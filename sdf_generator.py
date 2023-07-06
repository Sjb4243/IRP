import pandas as pd
import subprocess
import os
import signal
#Requires a file with canonical smiles, the score and the chembl_ID
data = pd.read_csv("testing_merge.csv")
data = data[data["Score"] >= 7]

for index, smiles in enumerate(data["Name"]):
    chembl_id = data["chembl_id"].iloc[index]
    score = data["Score"].iloc[index]
    score = str(round(score, ndigits=3))
    filename = "/home/sjb176/IRP/chembl_sdfs/" + chembl_id + "_score_" + score + ".sdf"
    nom = chembl_id + "_score_" + score +  ".sdf"
    cmd = "obabel -:\'" + smiles + "\' -o sdf -O /home/sjb176/IRP/chembl_sdfs/" + chembl_id + "_score_" + score + ".sdf --gen3D -h"
    if nom not in os.listdir("/home/sjb176/IRP/chembl_sdfs/"):
        print(f"{index},{chembl_id}")
        try:
            test = subprocess.Popen(cmd, shell=True, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            test.wait(timeout=15)
            output, error = test.communicate()
        except subprocess.TimeoutExpired:
            os.remove("/home/sjb176/IRP/chembl_sdfs/" + chembl_id + "_score_" + score + ".sdf")
            print("Timeout")
            os.killpg(os.getpgid(test.pid), signal.SIGTERM)
            continue
        if not str(error).startswith("b'1 molecule"):
            print("Stereoerror")
            os.rename("/home/sjb176/IRP/chembl_sdfs/" + chembl_id + "_score_" + score + ".sdf",
                      "/home/sjb176/IRP/chembl_sdfs/" + chembl_id + "_score_" + score + "_error.sdf")
            filename = "/home/sjb176/IRP/chembl_sdfs/" + chembl_id + "_score_" + score + "_error.sdf"
    else:
        print(f"Skipping {chembl_id}")



