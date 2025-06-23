#@title ##run **ESMFold**
%%time
from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import torch
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
from scipy.special import softmax
import gc

def parse_output(output):
  pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
  plddt = output["plddt"][0,:,1]

  bins = np.append(0,np.linspace(2.3125,21.6875,63))
  sm_contacts = softmax(output["distogram_logits"],-1)[0]
  sm_contacts = sm_contacts[...,bins<8].sum(-1)
  xyz = output["positions"][-1,0,:,1]
  mask = output["atom37_atom_exists"][0,:,1] == 1
  o = {"pae":pae[mask,:][:,mask],
       "plddt":plddt[mask],
       "sm_contacts":sm_contacts[mask,:][:,mask],
       "xyz":xyz[mask]}
  return o

def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()
alphabet_list = list(ascii_uppercase+ascii_lowercase)
def run_mode(jobName, mySequence):
  jobname = "test" #@param {type:"string"}
  jobname = re.sub(r'\W+', '', jobname)[:50]

  sequence = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE" #@param {type:"string"}
  sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
  sequence = re.sub(":+",":",sequence)
  sequence = re.sub("^[:]+","",sequence)
  sequence = re.sub("[:]+$","",sequence)
  copies = 1 #@param {type:"integer"}
  if copies == "" or copies <= 0: copies = 1
  sequence = ":".join([sequence] * copies)
  num_recycles = 3 #@param ["0", "1", "2", "3", "6", "12", "24"] {type:"raw"}
  chain_linker = 25

  ID = jobname+"_"+get_hash(sequence)[:5]
  seqs = sequence.split(":")
  lengths = [len(s) for s in seqs]
  length = sum(lengths)
  print("length",length)

  u_seqs = list(set(seqs))
  if len(seqs) == 1: mode = "mono"
  elif len(u_seqs) == 1: mode = "homo"
  else: mode = "hetero"

  if "model" not in dir() or model_name != model_name_:
    if "model" in dir():
      # delete old model from memory
      del model
      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = torch.load(model_name, weights_only=False)
    model.eval().cuda().requires_grad_(False)
    model_name_ = model_name

  # optimized for Tesla T4
  if length > 700:
    model.set_chunk_size(64)
  else:
    model.set_chunk_size(128)

  torch.cuda.empty_cache()
  output = model.infer(sequence,
                      num_recycles=num_recycles,
                      chain_linker="X"*chain_linker,
                      residue_index_offset=512)

  pdb_str = model.output_to_pdb(output)[0]
  output = tree_map(lambda x: x.cpu().numpy(), output)
  ptm = output["ptm"][0]
  plddt = output["plddt"][0,...,1].mean()
  O = parse_output(output)
  print(f'ptm: {ptm:.3f} plddt: {plddt:.3f}')
  os.system(f"mkdir -p {ID}")
  prefix = f"{ID}/ptm{ptm:.3f}_r{num_recycles}_default"
  np.savetxt(f"{prefix}.pae.txt",O["pae"],"%.3f")
  with open(f"{prefix}.pdb","w") as out:
    out.write(pdb_str)

def run(fileName, drivePath):
  with open(fileName, "r") as allProtFile:
    temp_seq = []
    temp_seqname = ""
    for i in allProtFile.readlines():
      i = i.strip()
      if i.startswith(">"):
        if temp_seq:
          print(f"正在预测{temp_seqname}")
          mySequence = "".join(temp_seq)
          run_mode(temp_seqname, mySequence)
          print(f"{temp_seqname}预测完成")
          os.system(f"mv ./{ID}/*.pdb {drivePath}/{ID}.pdb")
        temp_seq.clear()
        temp_seqname = i.split()[0].replace(">", "")
      else:
        temp_seq.append(i)
    print(f"正在预测{temp_seqname}")
    mySequence = "".join(temp_seq)
    run_mode(temp_seq, mySequence)
    print(f"{temp_seqname}预测完成")
    os.system(f"mv ./{ID}/*.pdb {drivePath}/{ID}.pdb")
     

from google.colab import drive
drive.mount('/content/drive')
     

# 自己的蛋白质文件名
fileName = "AMG_covmode0_clustermode2_id0.3_c0.7.faa_rep_seq_short_1500.faa"
# 保存的云端文件夹路径
drivePath = "/content/drive/MyDrive"
run(fileName, drivePath)
