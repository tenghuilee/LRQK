import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2

from lrqk_attention import *

device = torch.device("cuda:0")
# %%
# force rewrite the attention class

model, tokenizer = load_lrqk_model(
    "Qwen/Qwen2.5-7B-Instruct",
    # "./llama_hf/llama-3-8b-instruct",
    # "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    device=device,
    # max_length=32000,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


question = """**Title**: *Marie Curie*
**Context**:
Marie Skłodowska Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different scientific fields—Physics and Chemistry. Born in Warsaw, Poland in 1867, Curie studied at the Flying University before moving to Paris to continue her studies at the Sorbonne. Together with her husband Pierre Curie, she discovered two radioactive elements: polonium and radium. Her work laid the foundation for the development of X-rays in surgery. During World War I, she developed mobile radiography units to provide X-ray services to field hospitals. Curie died in 1934 from aplastic anemia, a condition linked to prolonged exposure to radiation.

**Title**: *CRISPR-Cas9 Genome Editing*

**Context**:
CRISPR-Cas9 is a revolutionary gene-editing technology adapted from a natural defense mechanism found in *Streptococcus pyogenes* bacteria. In nature, bacteria use CRISPR sequences and Cas proteins to recognize and cut foreign DNA, such as that from viruses. In 2012, scientists Jennifer Doudna and Emmanuelle Charpentier demonstrated that this bacterial system could be repurposed to precisely edit the genomes of eukaryotic cells. The system relies on a guide RNA (gRNA) to direct the Cas9 nuclease to a specific DNA sequence, where it introduces a double-stranded break. The cell’s natural repair mechanisms—non-homologous end joining (NHEJ) or homology-directed repair (HDR)—can then be harnessed to disrupt or replace genes.

One major advantage of CRISPR-Cas9 over previous gene-editing methods like zinc-finger nucleases or TALENs is its programmability: researchers can simply design a short RNA sequence to target virtually any site in the genome. However, concerns remain about off-target effects, where the system cuts unintended genomic locations, potentially leading to mutagenesis. Clinical trials are currently underway to evaluate CRISPR-based therapies for genetic disorders such as sickle cell anemia and Leber congenital amaurosis.

---

## **The Development of Large Language Models: A Story of Human Ingenuity and Machine Learning**

### **1. The Genesis: Symbolic AI and the Early Days**

The journey toward Large Language Models began not with neural networks but with symbolic AI. In the 1950s and 1960s, researchers attempted to encode human knowledge into machines using hand-crafted rules. Projects like ELIZA, developed in 1966 by Joseph Weizenbaum at MIT, mimicked human conversation by manipulating symbols based on syntactic rules. Although groundbreaking for its time, symbolic AI quickly hit limitations—mainly, the brittleness of rule-based systems in the face of language’s ambiguity and variability.

### **2. The Emergence of Statistical NLP**

The 1980s and 1990s marked a shift from rule-based systems to statistical approaches. As computational power and access to textual data increased, researchers began to model language using probabilities. Techniques like **n-grams** became popular, allowing systems to predict the next word based on a fixed number of previous words. These models laid the groundwork for language modeling by recognizing language as a probabilistic process.

Corpora like the **Penn Treebank** enabled empirical evaluation, and algorithms such as Hidden Markov Models (HMMs) and Conditional Random Fields (CRFs) became staples for tasks like part-of-speech tagging and named entity recognition. However, these models lacked the ability to capture long-range dependencies in text.

### **3. The Deep Learning Revolution**

The real breakthrough came in the 2010s with the advent of **deep learning**. Neural networks, especially **Recurrent Neural Networks (RNNs)** and their improved variants like **Long Short-Term Memory (LSTM)** networks, enabled models to maintain context across sequences. Google’s 2014 **Seq2Seq** model showcased the potential of RNNs for tasks like machine translation.

However, RNNs were computationally intensive and still struggled with very long sequences. Enter the **Transformer** model, introduced by Vaswani et al. in the seminal 2017 paper “Attention is All You Need.” The Transformer revolutionized NLP by replacing recurrence with **self-attention mechanisms**, allowing models to weigh the importance of different words in a sentence more effectively and in parallel.

### **4. The Rise of Pretrained Language Models**

Transformers laid the foundation for **pretrained language models**, which could be trained on massive corpora and then fine-tuned for downstream tasks. The release of **BERT (Bidirectional Encoder Representations from Transformers)** by Google in 2018 was a major milestone. BERT’s bidirectional training allowed it to understand context from both directions, improving performance across a wide range of NLP benchmarks.

Soon after, OpenAI introduced **GPT (Generative Pretrained Transformer)**. Unlike BERT, which was optimized for classification tasks, GPT was designed as a generative model, capable of producing coherent and contextually appropriate text given a prompt. **GPT-2**, released in 2019, demonstrated unprecedented fluency and coherence, sparking both excitement and concern over its potential misuse.

### **5. Scaling Laws and the Age of LLMs**

One of the most pivotal discoveries in the development of LLMs was the existence of **scaling laws**—empirical relationships showing that model performance improves predictably with more parameters, data, and compute. This insight, championed by researchers at OpenAI, justified the push toward **ever-larger models**.

The release of **GPT-3** in 2020, with 175 billion parameters, brought LLMs into the mainstream. Capable of writing essays, coding, composing poetry, and even answering philosophical questions, GPT-3 blurred the lines between machine output and human creativity. It was followed by a flurry of large models including **T5**, **XLNet**, and **ERNIE**, each pushing the boundaries of scale and generalization.

### **6. Instruction Tuning and Chat Models**

Despite their impressive capabilities, early LLMs lacked direction—they generated plausible text but often ignored user intent. This limitation was addressed by techniques like **instruction tuning** and **reinforcement learning from human feedback (RLHF)**. Models like **InstructGPT** and **ChatGPT** were trained not just to generate language, but to follow instructions and align with human preferences.

This paradigm shift turned LLMs from impressive demos into practical tools. Enterprises began integrating them into workflows for summarization, data extraction, and conversational agents.

### **7. Open Models and Democratization**

While companies like OpenAI and Google pushed the envelope with proprietary models, open-source initiatives gained momentum. Models like **GPT-Neo**, **GPT-J**, and **LLaMA** demonstrated that high-performing LLMs could be trained and released by independent groups and academic consortia. This democratization spurred innovation in training efficiency, model interpretability, and ethical use.

### **8. Challenges and Frontiers**

Despite their successes, LLMs face numerous challenges:

* **Bias and fairness**: LLMs can reproduce and amplify harmful stereotypes present in their training data.
* **Factual accuracy**: They often "hallucinate" plausible-sounding but incorrect information.
* **Energy consumption**: Training large models demands immense computational resources, raising environmental and economic concerns.
* **Alignment**: Ensuring models behave safely and predictably remains a grand challenge, especially as capabilities grow.

### **9. The Future: Toward AGI or Specialized Agents?**

As we look ahead, two trajectories emerge. One envisions **Artificial General Intelligence (AGI)**—models with general, human-level reasoning abilities. The other envisions **modular systems**, where LLMs are part of broader architectures that include planning, memory, and symbolic reasoning.

Ongoing work in **multimodal models** (combining text, image, audio, and video) and **tool-use** (integrating APIs and external reasoning modules) points toward increasingly capable and interactive systems.

---

## **Conclusion**

The development of LLMs represents one of the most remarkable stories in artificial intelligence—one that blends mathematics, engineering, linguistics, and philosophy. From humble rule-based beginnings to today’s sophisticated, billion-parameter behemoths, LLMs have reshaped our interaction with machines and challenged our understanding of language, intelligence, and creativity. The journey is far from over, and each new model brings us a step closer to unraveling the mysteries of cognition itself.

### **Question**:

<!-- Why is CRISPR-Cas9 considered more versatile than zinc-finger nucleases or TALENs? --!>

What is the paper talk about?
"""


inputs = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant. You can solve the question quick and efficiently."},
        {"role": "user", "content": question},
    ],
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    max_length=41000,
    truncation=True,
    padding=True,
)

inputs = inputs.to(model.device)

print(inputs.input_ids.shape)

streamer = transformers.TextStreamer(tokenizer, skip_prompt=True)

num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads

with torch.no_grad():
    in_seq_len = inputs.input_ids.shape[1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            past_key_values=DynamicLRQKCache(
                num_key_value_groups=num_key_value_groups,
                r=16,
                num_active_tokens=1024,
                lite_tokens=32,
                max_iter=2,
                tol=0.01,
            ),
            do_sample=True,
            streamer=streamer if inputs.input_ids.shape[0] == 1 else None,
        )
