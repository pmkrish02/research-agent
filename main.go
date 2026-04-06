package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"github.com/google/uuid"
	"google.golang.org/genai"
)

func cosineSim(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

type wikiArticle struct {
	Name       string
	Content    string
	Similarity float64
}

func CompileHandler(client *genai.Client) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		files, err := os.ReadDir("./data/raw")
		if err != nil {
			http.Error(w, fmt.Sprintf("could not read raw dir: %v", err), http.StatusInternalServerError)
			return
		}

		wikiDir := filepath.Join(".", "data", "wiki")
		err = os.MkdirAll(wikiDir, 0755)
		if err != nil {
			http.Error(w, "could not create wiki dir", http.StatusInternalServerError)
			return
		}

		var compiled []string
		var errors []string

		for _, file := range files {
			if file.IsDir() {
				continue
			}
			rawPath := filepath.Join(".", "data", "raw", file.Name())
			data, err := os.ReadFile(rawPath)
			if err != nil {
				errors = append(errors, fmt.Sprintf("read %s: %v", file.Name(), err))
				continue
			}

			result, err := client.Models.GenerateContent(
				r.Context(),
				"gemini-3.1-flash-lite-preview",
				genai.Text("Compile this raw text into a structured wiki article. Include key concepts, definitions, and relationships:\n\n"+string(data)),
				nil,
			)
			if err != nil {
				errors = append(errors, fmt.Sprintf("compile %s: %v", file.Name(), err))
				continue
			}

			wikiText := result.Text()
			wikiPath := filepath.Join(wikiDir, file.Name())
			err = os.WriteFile(wikiPath, []byte(wikiText), 0644)
			if err != nil {
				errors = append(errors, fmt.Sprintf("write %s: %v", file.Name(), err))
				continue
			}
			compiled = append(compiled, file.Name())
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"status":   "compiled",
			"compiled": compiled,
			"errors":   errors,
		})
	}
}

func ResearchHandler(client *genai.Client) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Question string `json:"question"`
		}
		err := json.NewDecoder(r.Body).Decode(&req)
		if err != nil {
			http.Error(w, "bad JSON", http.StatusBadRequest)
			return
		}

		qContent := []*genai.Content{genai.NewContentFromText(req.Question, genai.RoleUser)}
		qEmbed, err := client.Models.EmbedContent(r.Context(), "gemini-embedding-001", qContent, nil)
		if err != nil {
			http.Error(w, fmt.Sprintf("embed error: %v", err), http.StatusInternalServerError)
			return
		}
		questionVec := qEmbed.Embeddings[0].Values

		files, err := os.ReadDir("./data/wiki")
		if err != nil {
			http.Error(w, "no wiki found, run /compile first", http.StatusBadRequest)
			return
		}

		var articles []wikiArticle
		for _, file := range files {
			if file.IsDir() {
				continue
			}
			data, err := os.ReadFile(filepath.Join(".", "data", "wiki", file.Name()))
			if err != nil {
				continue
			}

			aContent := []*genai.Content{genai.NewContentFromText(string(data), genai.RoleUser)}
			aEmbed, err := client.Models.EmbedContent(r.Context(), "gemini-embedding-001", aContent, nil)
			if err != nil {
				continue
			}

			sim := cosineSim(questionVec, aEmbed.Embeddings[0].Values)
			articles = append(articles, wikiArticle{
				Name:       file.Name(),
				Content:    string(data),
				Similarity: sim,
			})
		}

		sort.Slice(articles, func(i, j int) bool {
			return articles[i].Similarity > articles[j].Similarity
		})

		tokenBudget := 2000
		var selected []wikiArticle
		currentTokens := 0
		for _, a := range articles {
			articleTokens := len(a.Content) / 4
			if currentTokens+articleTokens > tokenBudget {
				if len(selected) == 0 {
					truncLen := tokenBudget * 4
					if truncLen > len(a.Content) {
						truncLen = len(a.Content)
					}
					a.Content = a.Content[:truncLen]
					selected = append(selected, a)
				}
				break
			}
			selected = append(selected, a)
			currentTokens += articleTokens
		}

		var contextParts []string
		var sourceNames []string
		for _, a := range selected {
			contextParts = append(contextParts, a.Content)
			sourceNames = append(sourceNames, a.Name)
		}

		wikiContext := strings.Join(contextParts, "\n---\n")
		tokenEstimate := len(wikiContext) / 4

		decomposePrompt := fmt.Sprintf("Break this research question into 2-3 focused sub-questions. Return ONLY the sub-questions, one per line:\n\n%s", req.Question)
		decompResult, err := client.Models.GenerateContent(
			r.Context(),
			"gemini-3.1-flash-lite-preview",
			genai.Text(decomposePrompt),
			nil,
		)
		if err != nil {
			http.Error(w, fmt.Sprintf("decompose error: %v", err), http.StatusInternalServerError)
			return
		}
		subQuestions := strings.Split(strings.TrimSpace(decompResult.Text()), "\n")

		prompt := fmt.Sprintf("Based on this knowledge base:\n%s\n\nAnswer this research question and its sub-questions:\nMain question: %s\nSub-questions:\n%s",
			wikiContext, req.Question, strings.Join(subQuestions, "\n"))

		result, err := client.Models.GenerateContent(
			r.Context(),
			"gemini-3.1-flash-lite-preview",
			genai.Text(prompt),
			nil,
		)
		if err != nil {
			http.Error(w, fmt.Sprintf("LLM error: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"question":      req.Question,
			"sub_questions":  subQuestions,
			"answer":        result.Text(),
			"sources_used":  sourceNames,
			"tokens_used":   tokenEstimate,
			"token_budget":  tokenBudget,
			"within_budget": tokenEstimate <= tokenBudget,
		})
	}
}

func IngestHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Text string `json:"text"`
	}
	err := json.NewDecoder(r.Body).Decode(&req)
	if err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}
	newuuid := uuid.NewString()
	userFileDir := filepath.Join(".", "data", "raw")
	userFileName := newuuid + ".txt"
	userFilePath := filepath.Join(userFileDir, userFileName)
	err = os.MkdirAll(userFileDir, 0755)
	if err != nil {
		http.Error(w, "could not create directory", http.StatusInternalServerError)
		return
	}
	err = os.WriteFile(userFilePath, []byte(req.Text), 0644)
	if err != nil {
		http.Error(w, "could not save file", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"id": newuuid, "status": "ingested"})
}

func HealthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func main() {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, nil)
	if err != nil {
		log.Fatal("Could not create Gemini client:", err)
	}

	router := http.NewServeMux()
	router.HandleFunc("GET /health", HealthHandler)
	router.HandleFunc("POST /ingest", IngestHandler)
	router.HandleFunc("POST /compile", CompileHandler(client))
	router.HandleFunc("POST /research", ResearchHandler(client))

	fmt.Println("Research Agent running on :8080")
	log.Fatal(http.ListenAndServe(":8080", router))
}