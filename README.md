# SPAR 2024. Trajectories of LLMs. Mech Interp

This is a project lead by Nicky Pochinkov, carried out by Lovkush Agarwal and Angelo Benoit.

High-level question is: given an LLM (and its weights) and the activations on some sequence of tokens `t1,...,tn`, what can we predict about the LLM's outputs for future tokens `t_{n+1},...,`.

## Summary of work

So far, we have focussed on one particular special case.
Suppose LLM is given task of writing a blogpost about a recipe, with the start of blogpost being story that inspired the recipe.
We can ask: how determined is the recipe given the story?

Lovkush so far focussed on generating examples and just empirically seeing how much the recipe can vary given a starting text for the blogpost.
In particular, looking at how much the recipe can vary as you increase the temperature.

Angelo is adapting ideas from FutureLens paper, doing some kind of patching to see how much information is stored in the activations of the `\n` token between the story and the recipe.

## Summary of notebooks

- 01_la_first_experiments. LA just trying to generate text using transformers lens.
Without quantization, could not use the medium sized models on 16GB GPU that is available to us.
- 02_ab_playground
- 03_la_quantization. LA using hugging face interface to generate text with quantization.
Includes first development of visuals of the generated text and use of visuals on 'Agnes' scenario.
- 04_la_different_starts. LA going through process from 03_la notebook but with several different scenarios.
- 05_la_change_words_village_scenario. For one of the scenarios that named a particular French village, changed the name to see what qualitatively changes.
- 06_la_one_scenario_repeated. LA repeated one of the scenarios several times to see how much the CDF plots vary just due to noise/randomness.