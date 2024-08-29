# Notes on transferring longer topics

- Topic 1 always axe-throwing.
- Topic 2 some 'longer' topic.
- Looped through topic 2, created prompt to talk about each topic, generated once, split by the first new line (producing original prompt and original output), did transfer with `original prompt + \n\n` and generated 3 outputs.

## Thoughts on individual examples

Standard examples. Transferred outputs usually capture part of the topic, not whole topic.

- Ancient Greek philosophy. All outputs about Ancient Greek philosophERS instead
- Sustainable architecture practices. All outputs about sustainable practices
- Deep sea creatures. Starts with deep sea or oceans. Two mention creatures after several words.
- Genetic engineering ethics. Outputs about genetic engineering.
- 18th century literature. One starts with 18th century literature. Other two about world of literature
- Quantum computing applications. About quantum computing
- African wildlife conservation. Twice had african wildlife facing multiple threats. Once about African elephants.
- Industrial revolution impacts. About industrial revolution
- Space exploration milestones. All three start with 'Journey of a thousand miles begins with a single step'. Two move to space exploration. One just keeps repeating the saying...
- Artificial intelligence development. nothing noteworthy.
- Ancient Egyptian hieroglyphs. One got it, two about Ancient Egyptians.
- Climate change effects. All start with Climate change and mention effects.
- Psychological research methods. 1 psychology, 1 sociology, 1 social psychology
- Modern dance techniques. all about modern dance.
- Microbrewery beer production. 2 about brewing beer. 1 about microbreweries.
- Sustainable fashion trends. All about sustainable fashion
- Coral reef ecosystems. Coral reefs (being under threat)
- Cybersecurity best practices. Cybersecurity
- Classical music composers. Classical music
- World War II strategies. WWII
- Virtual reality applications. VR
- Indigenous art forms. Indigenous communities.
- Astrophysics recent findings. All start with 'Recent' and move to advancements or research in AI
- Urban planning challenges. 1 Urban planning, 1 about challenge of rapid growth in urban planning, 1 about urban sprawl as a problem.
- Culinary fusion experiments. Culinary innovation or creativity
- Cryptocurrency market fluctuations. All are about crypcurrency market volatility and price fluctuations.
- Bird migration patterns. Bird watching.
- 3d printing innovations. 3d printing.
- Sign language variations. All begin 'Signaling the end of the era, the iconic...'
- Nanotechnology medical applications. Nanotechnology.
- Mindfulness meditation benefits. All start 'Mindfulness' without meditation and go on to discuss benefits.
- Forensic science techniques. Forensic science.
- Environmentally friendly transportation. 2 match, 1 about environmentally friendly practices.
- Alternative energy storage. Alternative energy sources.
- Global trade agreements. Global trade.
- Sustainable water management. Sustainable practices or sustainable agriculture.

Axe-throwing had multiple paragraphs. This messed up my basic logic to identify the change of topic (which was searching for first new line) and so I was transferring from the wrong newline token.

- Tropical fruit varieties
- Electric car technology
- Jazz music history
- Medieval warfare tactics
- Volcanic eruption patterns
- Impressionist painting movement
- Endangered language perservation
- Neuroscience breakthrough discoveries

Unusual and notable examples

- Traditional Japanese cuisine. This has multiple paragraphs for topic1. HOWEVER, the completion from the newline token had traditional japanese cuisine in all 3 completions!!
  - Blockchain technology uses. Same thing - newline was in middle of axe throwing paragraphs but transferred generation was not about axe throwing. Was about blockchain technology, (but not its uses).
  - Extreme sports safety. Same thing again. Generations all start with 'Extreme', two about extreme weather, one about extreme heat warnings

- Renewable energy sources. All three started with a disclaimer about information should not be substituted for professional advice. And two then talked about renewable energy sources! Third about AI in healthcare.

- Organic farming methods. Original output started 'World of organic farming'. Two of the transferred outputs start 'World of professional wrestling'. But third generation started with something totally random: 'Process of making a good cup of coffee'
  - Turns out if you search for 'Process of making' this turns up several other times and always in the examples where axe-throwing occupied multiple paragraphs and so I was transferring incorrect newline.

## Code used

```python
for topic2 in topics_long[1:]:
    prompt_template = "Tell me about {topic1} in 150 words and then tell me about {topic2} in another 150 words. Only do that. Make sure you don't add any headings or comments.\n\n"
    topic1 = "axe-throwing"
    prompt = prompt_template.format(topic1=topic1, topic2=topic2)
    first_output = m.generate(prompt, 400)
    start = first_output[1][:first_output[1].find('\n')] + '\n\n'
    prompt_original = prompt + start
    output_original = first_output[1][first_output[1].find('\n')+2:]
    
    n_tokens_to_transfer = 1
    prompt_new, token_index_map, tokens_original, tokens_new = create_new_prompt_from_end_tokens(
        m=m, prompt_original=prompt_original, n_tokens_to_transfer=n_tokens_to_transfer, prefix=""
    )
    
    for h in m.hooks.neuron_replace.values():
        h.reset()
    
    activations_original = m.get_midlayer_activations(prompt_original)

    for original_index, new_index in token_index_map.items():
        for layer_type in ["mlp", "attn"]:
            for layer_number in range(m.cfg.n_layers):
                hook = m.hooks.neuron_replace[f"layer_{layer_number}_{layer_type}_pre_out"]
                hook.add_token(
                    new_index,
                    activations_original[layer_type][0, layer_number, original_index],
                )
    
    current_time = "2024-08-29_08-29-11"
    filename = f"../results/{current_time}_LA_activation_transfer_long_topics.jsonl"

    max_new_tokens = 150
    temperature = 0.2

    with HiddenPrints():
        for i in range(3):
            output = m.generate(prompt_new, max_new_tokens, temperature=temperature)

            data = {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "model": model_name,
                "transplant_layers": (0, m.cfg.n_layers),
                "transferred_token_num": n_tokens_to_transfer,
                "orig_prompt": prompt_original,
                "orig_output": output_original,
                "transplant_prompt": prompt_new,
                "other_info": f"gemma-{topic1}-{topic2}",
                "output": output[1],
            }

            with open(filename, "a") as file:
                file.write(json.dumps(data) + "\n")
```
