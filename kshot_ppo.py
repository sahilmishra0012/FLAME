import collections
import os
import json
import random
import pandas as pd
import os
from peft import LoraConfig, TaskType  # type: ignore
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
import wandb
import networkx as nx
from rank_bm25 import *
from sklearn.cluster import KMeans
from trl import AutoModelForCausalLMWithValueHead

from datasets import Dataset
# from trl import SFTTrainer
from transformers import (LlamaTokenizerFast,
                          TrainingArguments,
                          BertTokenizer,
                          BertModel)
from transformers import Adafactor

from trl import PPOConfig


from trl import PPOTrainer
from tqdm import tqdm

import torch

import math

import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity

import spacy

wandb.init()

os.environ["http_proxy"] = "http://proxy61.iitd.ac.in:3128"
os.environ["https_proxy"] = "http://proxy61.iitd.ac.in:3128"

model_path = "llama2-7b-hf/"
tokenizer_path = "llama2-7b-hf/"

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                         inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.4)

with open(os.path.join("./data/environment/environment_raw_en.taxo")) as f:
    taxonomy = f.readlines()


concept_set = set([])
all_taxo_dict = collections.defaultdict(list)
for pair in taxonomy:
    _, child, parent = pair.split("\n")[0].split("\t")
    concept_set.add(parent.strip().lower())
    concept_set.add(child.strip().lower())

with open("./data/environment/environment_train.taxo") as f:
    train_taxonomy = f.readlines()

train_edges = []
train_concept_set = set([])
taxonomy_edges = []
for line in train_taxonomy:
    parent, child = line.strip().split("\t")
    train_concept_set.add(parent.strip().lower())
    train_concept_set.add(child.strip().lower())
    taxonomy_edges.append((parent.strip().lower(), child.strip().lower()))


f = open('./data/environment/env_dict.json')
definitions = json.load(f)
f.close()

defi = {}
for i in definitions.keys():
    defi[i.strip().lower()] = [definitions[i][0].strip().lower()]
definitions = defi

with open("./data/environment/environment_eval.gt") as f:
    eval_parents = f.readlines()

with open("./data/environment/environment_eval.terms") as f:
    eval_children = f.readlines()

class TaxStruct(nx.DiGraph):
    def __init__(self, edges):
        super().__init__(edges)
        self.check_useless_edge()
        self._root = ""
        for node in self.nodes:  # find root
            if self.in_degree(node) == 0:
                self._root = node
                break
        assert self._root != ""

    def check_useless_edge(self):
        """
        delete useless edges
        """
        bad_edges = []
        for node in self.nodes:
            if len(self.pred[node]) <= 1:
                continue
            for pre in self.predecessors(node):
                for ppre in self.predecessors(node):
                    if ppre != pre:
                        if nx.has_path(self, pre, ppre):
                            bad_edges.append((pre, node))
        self.remove_edges_from(bad_edges)


taxonomic_tree = TaxStruct(taxonomy_edges)

bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


lis = []
for i in train_concept_set:
    defi = definitions[i.lower()][0]
    encoded_input = bert_tokenizer(i, defi, return_tensors='pt')
    output_embeddings = bert_model(**encoded_input)  # type: ignore
    lis.append(output_embeddings["last_hidden_state"][0][0].detach().numpy())

objective_function = []
for i in range(1, 25):
    clustering = KMeans(n_clusters=i, n_init=10)
    clustering.fit(lis)
    objective_function.append(clustering.inertia_)


kmeans = KMeans(n_clusters=10, n_init=10)
clusters = kmeans.fit_predict(lis)


node_to_cluster = {}
for i, j in zip(train_concept_set, clusters):
    node_to_cluster[i] = j


cluster_to_node = collections.defaultdict(list)
for i, j in zip(train_concept_set, clusters):
    cluster_to_node[j].append(i)

def get_local_pool(query, top_k=5, def_required=True):
    cluster_id = node_to_cluster[query]
    if def_required == True:
        tokenized_corpus = [(doc + " -> " + definitions[doc.lower()][0]
                             ).lower().split(" ") for doc in cluster_to_node[cluster_id]]
        tokenized_query = (
            query + " -> " + definitions[query.lower()][0]).lower().split(" ")
    else:
        tokenized_corpus = [(doc).lower().split(" ")
                            for doc in cluster_to_node[cluster_id]]
        tokenized_query = (query).lower().split(" ")
    bm25 = BM25Plus(tokenized_corpus)
    docs = bm25.get_top_n(tokenized_query, tokenized_corpus, n=top_k)
    if def_required == True:
        docz = []
        for i in docs:
            docz.append(" ".join(i).split("->")[0].strip())
        return docz
    return docs


def get_global_pool():
    global_pool_count = 5
    global_pool = []
    clusters = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(global_pool_count):
        cluster_id = random.choice(clusters)
        node = random.choice(cluster_to_node[cluster_id])
        global_pool.append(node.lower().strip())
    return global_pool

instruction = "You are an assistant to hypernym prediction and sorting.\nGiven a term, its context and a list of candidate hypernym answers to this term, You need to rank these candidate terms in the list to let candidate answers which are most possible to be the hypernym or parent term to the given term and return the list.\nCandidate Hypernym List = [polluted area, underwater mineral resources, animal life, environment, EU environmental policy, environmental statistics, quality of the environment, rodent, Adriatic Sea, emission allowance, polluter pays principle, acidification, whale, biodiversity, sea-bed, atmosphere, noise level, resources of the sea, ecological balance, coastal protection, marine mammal, wild mammal, degradation of the environment, environmental law, marsupial, waste management, Baltic Sea, degree of pollution, saltwater, environmental tax, climatic zone, island, Indian Ocean, chemical waste, eutrophication, motor vehicle pollution, radioactive pollution, Black Sea, natural hazard, protection of animals, ozone, surface water, Irish Sea, pollution control measures, estuary, environmental monitoring, greenhouse gas, tradeable emission permit, underground storage of waste, ecosystem, bathing water, Arctic, environmental indicator, noise protection, conservation of resources, national city park, joint implementation, Norwegian Sea, Red Sea, risk prevention, English Channel, insect, national park, equatorial zone, erosion, seal, fire protection, nature reserve, renewable resources, pollutant, EU emission allowance, waste recycling, bear, earthquake, export of waste, oil pollution, exploitation of the sea-bed, frigid zone, fire, littoral, deposit on a polluting product, aquatic environment, marine life, water management, thermal discharge, anti-pollution device, arid zone, environmental education, fight against wastage, industrial hazard, Antarctic Ocean, environmental policy, protected area, pollution, noise, fight against insects, physical environment, Mediterranean Sea, Ionian Sea, polar region, marine pollution, plant life, deforestation, biodegradability, natural disaster, metal waste, ocean, dolphin, dangerous substance, lake, climate, humid zone, fur-bearing animal, lynx, nuisance, animal resources, used oil, energy resources, freshwater, Caspian Sea, water protection, prevention of pollution, pollution from ships, defoliation, pollution control, combustion gases, toxic substance, mountain, protection of animal life, drinking water, protection of plant life, coastal pollution, industrial pollution, economic instrument for the environment, management of resources, decontamination, cost of pollution, global warming, atmospheric pollution, environmental protection, water resources, pollution from agricultural sources, cyclone, wastewater, natural resources, domestic waste, atmospheric conditions, radioactive waste, agricultural disaster, hospital waste, geophysical environment, electromagnetic interference, watercourse, climate change, bad weather, emission trading, greenhouse effect, thermal pollution, water pollution, pollution of waterways, biotope, seismic monitoring, corrosion, Atlantic Ocean, organic pollution, Aegean Sea, exploitation of the seas, chemical pollution, dumping of waste, environmental standard, unauthorised dumping, non-recoverable waste, noise pollution, drought, atmospheric pollutant, metal pollution, storage of waste, environmental economics, countryside conservation, sea, bird, volcanic eruption, climate change policy, waste disposal, electronic waste, stratospheric pollutant, environmental research, soil pollution, protected species, adaptation to climate change, sewage sludge, water, exploitation of resources, stratospheric pollution, mineral resources, marine environment, man-made disaster, waste, sensitive area, harmful plant, pollution from land-based sources, local pollution, Pacific Ocean, Arctic Ocean, biosphere, wildlife, plant resources, destruction of crops, non-ionising radiation]\nA few examples of hypernym-hyponym are given as:\n"
k_shot_inputz = []
labelz = []
k_shot_outputz = []
for i in train_concept_set:
    if i == "environment":
        continue

    local_pool = get_local_pool(i)
    global_pool = get_global_pool()

    local_k_shots = ""
    global_k_shots = ""
    for j in local_pool[1:3]:
        if j == "environment":
            continue
        local_shot = "TERM: {}\nCONTEXT: {}\nHYPERNYM: {}\n".format(
            j, definitions[j][0], [pred for pred in taxonomic_tree.predecessors(j)][0])
        local_k_shots += local_shot

    for j in global_pool[:3]:
        if j == "environment":
            continue
        global_shot = "TERM: {}\nCONTEXT: {}\nHYPERNYM: {}\n".format(
            j, definitions[j][0], [pred for pred in taxonomic_tree.predecessors(j)][0])
        global_k_shots += global_shot

    label = [pred for pred in taxonomic_tree.predecessors(i)][0]
    ques = "TERM: {}\nCONTEXT: {}\nHYPERNYM:".format(
        i, definitions[i][0])
    prompt = instruction + local_k_shots + global_k_shots + ques
    k_shot_inputz.append(prompt)
    labelz.append(label)
    k_shot_outputz.append(prompt + " " + label)
    local_k_shots = ""
    global_k_shots = ""
    for j in local_pool[3:]:
        if j == "environment":
            continue
        local_shot = "TERM: {}\nCONTEXT: {}\nHYPERNYM: {}\n".format(
            j, definitions[j][0], [pred for pred in taxonomic_tree.predecessors(j)][0])
        local_k_shots += local_shot
    for j in global_pool[2:]:
        if j == "environment":
            continue
        global_shot = "TERM: {}\nCONTEXT: {}\nHYPERNYM: {}\n".format(
            j, definitions[j][0], [pred for pred in taxonomic_tree.predecessors(j)][0])
        global_k_shots += global_shot

    label = [pred for pred in taxonomic_tree.predecessors(i)][0]
    ques = "TERM: {}\nCONTEXT: {}\nHYPERNYM:".format(
        i, definitions[i][0])
    prompt = instruction + local_k_shots + global_k_shots + ques
    k_shot_inputz.append(prompt)
    labelz.append(label)
    k_shot_outputz.append(prompt + " " + label)

data = pd.DataFrame({"input": k_shot_inputz, "label": labelz})

train_data = Dataset.from_pandas(data)


config = PPOConfig(
    batch_size=32,
    model_name=model_path,
    learning_rate=1.41e-5,
    log_with="wandb"
)

peft_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": 0},
    peft_config=peft_config,
)

sum(p.numel() for p in peft_model.parameters())

sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path, is_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["input"])
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample

dataset = train_data.map(tokenize, batched=False)
dataset.set_format(type="torch")

optimizer = Adafactor(
        filter(lambda p: p.requires_grad, peft_model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

ppo_trainer = PPOTrainer(
    model=peft_model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
    optimizer=optimizer,
    data_collator = collator
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}


nlp = spacy.load('en_core_web_sm')

class Reward:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def len_diff(self, i, j):
        length = abs(len(i)-len(j))
        if length==0:
            return 1.0
        else:
            return -length/(len(i)+len(j))
    
    def match(self, i, j):
        if i.lower()==j.lower():
            return 1.0
        else:
            return 0.0
    
    def word_count(self, i, j):
        diff = abs(len(i.split(" "))-len(j.split(" ")))
        if diff==0:
            return 1.0
        else:
            return -diff*1.0
    
    def cosine(self, i, j):
        embedding1 = nlp(i.lower()).vector
        embedding2 = nlp(j.lower()).vector
        embedding1 = bert_model(**bert_tokenizer(i.lower(), return_tensors='pt'))
        embedding2 = bert_model(**bert_tokenizer(j.lower(), return_tensors='pt'))
        similarity_score = cosine_similarity(embedding1["last_hidden_state"][0][0].detach().numpy().reshape(1, -1), embedding2["last_hidden_state"][0][0].detach().numpy().reshape(1, -1))[0][0]
        return similarity_score

    def fuzz_match(self, i, j):
        ratio = fuzz.ratio(i.lower(), j.lower())
        partial_ratio = fuzz.partial_ratio(i.lower(), j.lower())
        token_sort_ratio = fuzz.token_sort_ratio(i.lower(), j.lower())
        token_set_ratio = fuzz.token_set_ratio(i.lower(), j.lower())
        return (ratio + partial_ratio + token_sort_ratio + token_set_ratio)/400.0

    def get_rewards(self, batch):
        rewards = []
        for i, j in zip(batch["response"], batch["label"]):
            true = j.strip()
            pred = i.split("\n")[0].strip()
            reward1 = self.len_diff(true, pred)
            reward2 = self.fuzz_match(true, pred)
            reward3 = self.cosine(true, pred)
            reward4 = self.match(true, pred)
            reward5 = self.word_count(true, pred)
            rewards.append(torch.tensor((reward1 + reward2 + reward3 + reward4 + reward5)/1.0))
        return rewards

reward = Reward(16)

for epoch in tqdm(range(0,100)):
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]
        response = ppo_trainer.generate(query_tensors, return_prompt=False, max_new_tokens=8, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=False) for r in response]
        score = reward.get_rewards(batch)
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response, score)
        ppo_trainer.log_stats(stats, batch, score)
    ppo_trainer.model.save_pretrained("trained-model")
    print("saved model for epoch", epoch)


















