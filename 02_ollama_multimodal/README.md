# AI Powered apps in Golang with Genkit -- Ollama, Vision and Inference with Context (Part II)

In this blog entry we are going to continue exploring Genkit in Go. In Part I we saw how to use Genkit, the Genkit CLI and dev enviroment to launch an app that accepted user prompts for writing horror short paragraphs.

Today we are going to work with a vision based model and we will also add internal context to the prompt. We will use the Ollama plugin and the Llava model (instead of using the Google plugin and a Google model, like Gemini Flash 2.0, as we did in Part I).

Let's go!

# Deploying Ollama

Ollama is an open-source tool that allow users to run locally large language models (LLMs). You can download it from [the official Ollama web page](https://ollama.com/download). For linux and in case you have other types of GPU than NVIDIA, there are these [instructions](https://github.com/ollama/ollama/blob/main/docs/linux.md), which can be helpful.

Once you have it installed you can run `ollama serve` for running it locally. You can check if it has detected your GPU, because a line like the following will appear.

```
... level=INFO source=types.go:130 msg="inference compute" id=GPU-691ff71a-1458-f906-ea83-9642a50f70e7 library=cuda variant=v12 compute=6.1 driver=12.2 name="NVIDIA GeForce GTX 1080" total="7.9 GiB" available="7.8 GiB"
```

To verify that everything is working and ollama is accepting calls, you can run `ollama -v`, which should return the version you are running, in my case `ollama version is 0.7.0`

For the purposes of today's blog entry, we will pull the llava model and run it.

```
ollama pull llava
ollama run llava
```

Then, again from another terminal, and to verify that the communication to the running model is indeed working:

````
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "llava",
  "prompt": "Why is the sky blue?"
}'
````

And you should see a stream of JSON type messages each with a response and a boolean field named _done_, that indicates that the stream has finished or not. If you want to receive the full message directly instead of a stream you can add the field `"stream": false` to your query.

```
{"model":"llava","created_at":...,"response":" The blue color of the sky is due to a process called Rayleigh scattering. As sunlight passes through the Earth's atmosphere, it interacts with the molecules and particles in the air. Blue light has shorter wavelengths than other colors, such as red or orange, so it gets scattered more easily by these tiny particles. This scattering of blue light is what makes the sky appear blue to us.\n\nIn addition to Rayleigh scattering, there are also other factors that can affect the color of the sky, such as the time of day and the presence of pollution or clouds in the atmosphere. However, Rayleigh scattering is the primary reason why the sky appears blue to our eyes. ","done":true,"done_reason":"stop","context":[...],"total_duration":3340214270,"load_duration":4369854,"prompt_eval_count":14,"prompt_eval_duration":17087964,"eval_count":145,"eval_duration":3318329537}
```

In this case we have downloaded and run Llava model, but Ollama supports a wide variety of models ([full list here](https://ollama.com/library)), including DeepSeek, Qwen, Llama, Phi and others. To check that it was working we have ask our model to generate an answer to a question, but there are many more actions to interact with the ollama instance: creating models, listing them, generating embeddings with a particular model, ... All possible actions are detailed [here, in the API docs](https://github.com/ollama/ollama/blob/main/docs/api.md).


# Sending an image query to ollama through Genkit Go

But today's target is not Ollama itself, but how to work with images and context with LLMs in Genkit Go. Let's instantiate the Ollama plugin and Genkit with it:

````
ctx := context.Background()

ollamaPlugin := &ollama.Ollama{ServerAddress: "http://127.0.0.1:11434"}

g, err := genkit.Init(ctx, genkit.WithPlugins(ollamaPlugin))
if err != nil {
	log.Fatal(err)
}
````

Remember not to use localhost but rather `127.0.0.1` if not the error `unsupported protocol scheme "localhost"` might appear. Let's then add the llava model to Ollama to be able to make queries through Genkit

````
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
````

As you can see, we can specify the type of model we are setting and the model information where we have different options for its configuration:

- Multiturn: does the model support multiturn chats?
- SystemRole: does the model support system messages?
- Media: does the model accept multimodal inputs?
- Tools: does the model support function calling (tools)?

Now we are ready to get an image and send it to the model along a question. 

```
contentType, dataURI, err := getImage("wally.jpeg")
if err != nil {
	log.Fatalf("Error getting the image %v", err)
}
// Create a request with text and image
request := &ai.ModelRequest{
	Messages: []*ai.Message{
		{
			Role: ai.RoleUser,
			Content: []*ai.Part{
				ai.NewTextPart("What do you think about this animated character?"),
				ai.NewMediaPart(contentType, dataURI),
			},
		},
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
```

_NOTE_: For running this, you just need to have ollama running (`ollama serve`), and in another terminal then call your go program: `go run main.go`. No need in these examples for launching the Genkit Dev Env that we used in the previous blog entry.

Answer of the model without context, just with the image and prompt.

>  This is an illustration of the fictional character Wally Bumpers, from the "Where's Wally?" series. The character is depicted with his signature glasses, red and white striped shirt, blue shorts, and a red hat with white pompoms. He is known for his adventures in various locations around the world and for helping people find their missing items.

# Adding context

An important concept in LLM prompting is giving context in form of an image, sentence or any other type of input to condition its output. This will be the fundamental idea when we explore Retrieval Augmented Generation (RAG). Let's add some text context to our previos query:

```
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
}
```
This is the output when we added context about where the image came from.

>  The animated character depicted in the image appears to be an exaggerated, stylized representation of a man with glasses and a mustache. While the mustache is a classic feature associated with many well-known fictional characters, the glasses might not necessarily indicate that this character is evil, as they can also simply be a part of the character's design or serve a functional purpose in their appearance. Without more context about the character and its role within a story or series, it's difficult to make a definitive assessment about the character's intentions or nature.

In this case the context didn't serve to any other purpose other than include a reference to it in the model generated output (which indeed seems to disagree).

# Controlling the different parameters of the model

Finally, another possibility when querying a model would be to modify its creativity parameters. These parameters will modify the behaviour of the model: overall how much it can hallucinate. The parameters offered in Genkit are: 

- Temperature: a scaling factor applied to the next token likelihood before converting it to a probability. Values between 0.0 and 1.0 expand the differences between tokens: those that were probable will be even more, and those that weren't will be even more unlikely, thus the model will be less creative. Values higher than 1.0 will get the model more creative since it will make probabilities among tokens more homogeneous.
- TopP: controls the number of tokens to consider indirectly through probability. 1.0 means to consider all tokens and towards 0.0 reduces the list of tokens considered according to their probability.
- TopK: controls the number of tokens to consider too, but in this case through the direct number of tokens.

For modifying the parameters we can add the following to the model request:

````
	request := &ai.ModelRequest{
...
		Config: &ai.GenerationCommonConfig{
			Temperature: 2.0,
			TopK:        50,
			TopP:        0.5,
		},
	}
````

And the model's answers:

>  The image shows an animated character who resembles the main character from the "Wizard of Oz," but is styled in a way that seems more contemporary or cartoonish. He is wearing a red hat and has a beard, which are characteristic features of the Wizard of Oz. The character's glasses add to the whimsical and fantasy-like appearance. It appears to be designed with a friendly and approachable style, often associated with animated characters from children's media. However, as per the information provided, the glasses being described as "evil" suggests that this interpretation of the character is not in line with the typical positive portrayal of the Wizard in his original stories or appearances.

Which seems to be off the rail talking about the Wizard of Oz, obviously due to the high temperature value

## Conclusions

In this blog entry we have seen how to use Ollama with Genkit, how to do multimodal requests and adding context to them. Finally we have seen how the model creativity parameters can be tweaked to change the model's behaviour.

All code is available in this repository: 


## References
- [Ollama plugin](https://firebase.google.com/docs/genkit-go/plugins/ollama)
- [Ollama Github page](https://github.com/ollama/ollama)
- [Ollama API reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama reference sample](https://github.com/firebase/genkit/blob/main/go/samples/ollama-vision/main.go)
- [Context reference sample](https://github.com/firebase/genkit/blob/main/go/samples/basic-gemini-with-context/main.go)
- [Generating content with Genkit](https://firebase.google.com/docs/genkit-go/models)
- [Writing a Genkit model plugin](https://genkit.dev/go/docs/plugin-authoring-models/#model-definitions)
