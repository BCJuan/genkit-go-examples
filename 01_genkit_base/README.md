# AI Powered apps in Golang with Genkit

I have been learning more and more about LLMs (Large Language Models) and LVMs (Large Vision Models) and I was curious about ways of serving these models. Since I have also been learning Golang lately, I started searching for this type of frameworks that allow you to deploy these big DL models. So I saw Genkit had a Go API and decided to give it a go.

So, what is Genkit? Genkit is an open-source Typescript toolkit from the Firebase platform that allows you to build AI powered apps that can integrate in the Firebase environment (and it also has a Golang API in alpha). It offers a unified interface and integration for deploying big time models (Gemini, LLama, Mistral, ...) and the focus is on quick deployment to leave room for user experience.

At a first sight looks like a framework which is focused on Retrieval Augmented Generation (RAG) and Chat Systems and thus oriented to large models. They don't mention any other type of ML models.

In their own words:

> Whether you're building chatbots, intelligent agents, workflow automations, or recommendation systems, Genkit handles the complexity of AI integration so you can focus on creating incredible user experiences

## Getting Genkit

First of all, let's get Genkit installed in our system. For the Genkit CLI and UI we need to get Node

```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install 20
npm i -g genkit
``` 
Now this has installed the CLI. You will see in the tutorials that you can then init genkit with `genkit init` but I had to run

```
npx genkit-cli init
```

See [this Github issue for details](https://github.com/firebase/genkit/issues/1295) where people complain just about this.

Okay, so we are almost there. Now, since the first model we are going to use is Google's Gemini, we need to create an API access key so we can get these models into Genkit. Generate one in [this link](https://aistudio.google.com/app/apikey) and then export it as 

```
export GOOGLE_GENAI_API_KEY=<your API key>
```

Cool, ready to go!

## Launching a sample application (horror suggestions)

We are going to launch a small application that given a user input will write a terror short paragraph about that.

This next piece of code is inside `main.go`:

```
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

```

Some things to note on the previous code snippet:
- We first initiate Google AI, then define the flow, and finally launch Genkit. This is important, otherwise your flow won't be loaded.
- Flows are wrapped functions that allow you to set up a context and additional charactestistics for specific processes (like generating stories, describing images, generating code, ...). In this case, we are generating a flow for taking a user input and generating a horror paragraph about it.
- As you can see we are using Gemini 1.5 Flash, and setting a new text generation request based on user input.

Now, we are ready to launch it with ` npx genkit-cli start -- go run main.go`

Once we do that some services will be launched for us. Mainly the Telemetry API and the Genkit Dev UI. Let's check the latter.

## Inspecting the dev UI and interacting with the flow

As we will see in the following video, in the Genkit Dev UI we can see the different models available, flows, ... It let us interact with our flow too being able to call the model for getting horror based paragraph in this case based in the Alabama word.

![](genkit_flow_example.mp4)

The output of the model is 

> "The humid Alabama air hung heavy, thick with the scent of honeysuckle and something else… something metallic and faintly sweet.  The cicadas’ incessant drone was punctuated by the unsettling *thwack* of something unseen impacting the screen door.  Grandpappy’s stories about the things that crawled out from the Black Belt swamps at night felt less like folklore and more like a chilling prophecy as the shadows deepened, swallowing the porch light in a suffocating darkness that pulsed with an unseen, malevolent heartbeat.\n"

which I find cool and intriguing enough.

## Conclusions

Genkit is not the only possibility for setting up LLMs in AI powered apps. There are other choices such as [Langchain](https://www.langchain.com/)

## Bonus track

If you are 
```
ssh -L 10000:localhost:4000 blue@192.168.1.136 -p 11050
```

# References

- https://firebase.google.com/docs/genkit 
- https://firebase.google.com/docs/genkit-go/get-started-go#gemini-google-ai 
- https://developers.googleblog.com/en/introducing-genkit-for-go-build-scalable-ai-powered-apps-in-go/
- https://github.com/golang/example/tree/master/ragserver/ragserver-genkit
- https://go.dev/blog/llmpowered
- https://en.wikipedia.org/wiki/Retrieval-augmented_generation

