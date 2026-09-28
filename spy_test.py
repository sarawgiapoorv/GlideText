import local_llm, ai_brain

orig = local_llm.LocalLLMEngine._clean_model_output

def spy(text, raw_text=""):
    print("\nRAW MODEL REPLY:", repr(text))
    result = orig(text, raw_text)
    print("AFTER CLEANER:  ", repr(result))
    return result

local_llm.LocalLLMEngine._clean_model_output = staticmethod(spy)

brain = ai_brain.AIBrain()
sample = ("I don't know what the whisper flow tool is doing because it is not giving me "
          "the decide-on. Sometimes it just print the text directly out of the speech. "
          "I don't know whether it polishes it or not, but the output which I want is "
          "not corrected one. And sometimes the output looks like it is broken.")
print("FINAL:", repr(brain.polish(sample)))
