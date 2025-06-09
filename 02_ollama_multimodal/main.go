package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/ollama"
)

func getImage(imagePath string) (string, string, error) {
	if len(os.Args) > 1 {
		imagePath = os.Args[1]
	}

	// Check if image exists
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		return "", "", fmt.Errorf("image file not found: %s", imagePath)
	}

	// Read the image file
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		return "", "", fmt.Errorf("failed to read image file: %v", imagePath)
	}

	// Detect content type (MIME type) from the file's binary signature
	contentType := http.DetectContentType(imageData)

	// If content type is generic/unknown, try to infer from file extension
	if contentType == "application/octet-stream" {
		contentType = getContentTypeFromExtension(imagePath)
	}

	// Encode image to base64
	base64Image := base64.StdEncoding.EncodeToString(imageData)
	dataURI := fmt.Sprintf("data:%s;base64,%s", contentType, base64Image)
	return contentType, dataURI, nil
}

// getContentTypeFromExtension returns a MIME type based on file extension
func getContentTypeFromExtension(filename string) string {
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".png":
		return "image/png"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	case ".bmp":
		return "image/bmp"
	case ".svg":
		return "image/svg+xml"
	default:
		return "image/png" // Default fallback
	}
}

func main() {

	ctx := context.Background()

	ollamaPlugin := &ollama.Ollama{ServerAddress: "http://127.0.0.1:11434"}

	g, err := genkit.Init(ctx, genkit.WithPlugins(ollamaPlugin))
	if err != nil {
		log.Fatal(err)
	}
	modelName := "llava"
	model := ollamaPlugin.DefineModel(
		g,
		ollama.ModelDefinition{
			Name: modelName,
			Type: "generate", // "chat" or "generate"
		},
		&ai.ModelInfo{
			Supports: &ai.ModelSupports{
				Multiturn:  false,
				SystemRole: true,
				Tools:      false,
				Media:      true,
			},
		},
	)
	contentType, dataURI, err := getImage("wally.jpeg")
	if err != nil {
		log.Fatalf("Error getting the image %v", err)
	}
	// Create a request with text and image
	request := &ai.ModelRequest{
		Docs: []*ai.Document{
			{
				Content: []*ai.Part{
					ai.NewTextPart("Glasses are often a sign of evil people"),
				},
			},
		},
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					ai.NewTextPart("What do you think about this animated character?"),
					ai.NewMediaPart(contentType, dataURI),
				},
			},
		},
		Config: &ai.GenerationCommonConfig{
			Temperature: 2.0,
			TopK:        50,
			TopP:        0.5,
		},
	}
	// Call the model
	fmt.Printf("Sending request to %s model...\n", modelName)
	response, err := model.Generate(ctx, request, nil)
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	// Print the response
	fmt.Println("\nModel Response:")
	for _, part := range response.Message.Content {
		if part.IsText() {
			fmt.Println(part.Text)
		}
	}
}
