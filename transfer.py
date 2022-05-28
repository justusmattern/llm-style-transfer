from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
from torch import optim
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def read_file(in_file):
    texts = []
    with open(in_file, 'r') as f:
        for line in f:
            texts.append(line.replace('\n', ' '))
    
    return texts


def run(in_file, out_file, prompt_in, prompt_out, model, learning_rate)

    tokenizer = GPT2Tokenizer.from_pretrained(model)
    in_texts = read_file(in_file)
    sim_model = SentenceTransformer('paraphrase-albert-small-v2')


    for text in in_texts:

        model = GPT2LMHeadModel.from_pretrained(model)
        model.parallelize()
        optimizer = optim.Adam(model.parameters, lr = learning_rate)

        for i in range(10):
            tokenized_text = tokenizer([prompt_in+text], return_tensors='pt').input_ids.to('cuda:0')
            tokenized_prompt = tokenizer([prompt_in], return_tensors='pt').input_ids.to('cuda:0')
            loss = model(tokenized_text, labels=tokenized_text).loss - model(tokenized_prompt, labels=tokenized_text).loss
            loss.backward()
            optimizer.step()
            print('loss', loss)

            input_ids = tokenizer.encode(prompt_out, return_tensors='pt').to('cuda:0')

            sample_outputs = model.generate(
                            input_ids,
                            do_sample=False, 
                            max_length=int(len(tokenized_text[0])*2),  
                            top_p=0.8,
                            no_repeat_ngram_size=3,
                            num_return_sequences=0
                        )

            generated_texts = []
            
            for sample in sample_outputs:
                text = tokenizer.decode(sample, skip_special_tokens = True)
                generated_texts.append(text.replace(prompt_out, ''))

                gen_embeddings = sim_model.encode(generated_texts)
                orig_embedding = sim_model.encode([text])
                sim_scores = cos_sim(gen_embeddings, orig_embedding)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type = str)
    parser.add_argument('--out-file', type = str)
    parser.add_argument('--prompt-in', type=str)
    parser.add_argument('--prompt-out', type=str)
    parser.add_argument('--model', type=str, default='gpt2-xl')
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    args = parser.parse_args()

    run(in_file = args.in_file, out_file = args.out_file, prompt_in = args.prompt_in, prompt_out = args.prompt_out, model=args.model, learning_rate = args.learning_rate)