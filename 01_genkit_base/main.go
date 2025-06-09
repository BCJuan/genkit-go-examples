package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googleai"
)

// API KEY
// export GOOGLE_GENAI_API_KEY=AIzaSyDMoCS2o2AfErg9pCmmFRxNwxsPJjEkhVc

func main() {

	ctx := context.Background()

	if err := googleai.Init(ctx, &googleai.Config{}); err != nil {
		log.Fatal(err)
	}

	// Define a simple flow that prompts an LLM to generate menu suggestions.
	genkit.DefineFlow("shortTerrorFlow", func(ctx context.Context, input string) (string, error) {
		// The Google AI API provides access to several generative models. Here,
		// we specify gemini-1.5-flash.
		m := googleai.Model("gemini-1.5-flash")
		if m == nil {
			return "", errors.New("shortTerrorFlow: failed to find model")
		}

		// Construct a request and send it to the model API (Google AI).
		resp, err := m.Generate(ctx,
			ai.NewGenerateRequest(
				&ai.GenerationCommonConfig{Temperature: 1},
				ai.NewUserTextMessage(fmt.Sprintf(`Write a small terror-based paragraph themed on %s`, input))),
			nil)
		if err != nil {
			return "", err
		}

		// Handle the response from the model API. In this sample, we just
		// convert it to a string. but more complicated flows might coerce the
		// response into structured output or chain the response into another
		// LLM call.
		text := resp.Text()
		return text, nil
	})

	if err := genkit.Init(ctx, &genkit.Options{}); err != nil {
		log.Fatal(err)
	}

	<-ctx.Done()

}
