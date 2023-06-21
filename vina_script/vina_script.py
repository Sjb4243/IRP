from vina import Vina
import os
import numpy as np
from numpy import savetxt
import pandas as pd

def get_folder_contents(recep_folder, lig_folder):
    recep_list = [os.path.join(recep_folder, file) for file in os.listdir(recep_folder)]
    lig_list = [os.path.join(lig_folder, file) for file in os.listdir(lig_folder)]
    #have to sort to match the names at least for the x by x matric version
    recep_list.sort()
    lig_list.sort()
    return recep_list, lig_list

def initialise_matrix(receptor_names, ligand_names):
    #Adding one to both so I can use a column and row for the names of the receptor/compoud
    recept_length = len(receptor_names) + 1
    ligand_length = len(ligand_names) + 1
    matrix = [[0 for col in range(0,recept_length)] for row in range(0, ligand_length)]
    for index, name in enumerate(receptor_names, 1):
        matrix[index][0] = name
    for index, name in enumerate(ligand_names, 1):
        matrix[0][index] = name
    return matrix


def get_names(recep_files, lig_files):
    #I did not stop to ask whether I not I should before thinking whether or not i could
    recep_names = [os.path.normpath(file).split(os.path.sep)[-1].split(".")[0] for file in recep_files]
    lig_names = [os.path.normpath(file).split(os.path.sep)[-1].split(".")[0] for file in lig_files]
    return recep_names, lig_names


def run_vina(matrix, recep_files, lig_files, recep_names, lig_names):
    error_list = []
    for index, receptor in enumerate(zip(recep_files, recep_names), 1):
        for index2, ligand in enumerate(zip(lig_files, lig_names), 1):
            print(f"Docking {receptor[1]} and {ligand[1]}...")
            v.set_receptor(receptor[0])
            v.set_ligand_from_file(ligand[0])
            v.compute_vina_maps(center=[-17.8626, -4.41595, -15.0678], box_size=[13.89, 12.61, 13.09])
            v.dock(exhaustiveness=8, n_poses=5)
            energies = v.energies()
            energy_list = [row[0] for row in energies]
            min_energy = min(energy_list)
            matrix[index][index2] = min_energy
            if v.score()[0] != min_energy:
                print("\nSort error detected")
                error_list.append([receptor[1], ligand[1]])
            if index == index2:
                v.write_poses(receptor[1] + ligand[1] + ".pdbqt", n_poses=5, overwrite=True)
            print(f"Matrix position {index},{index2} being updated with {min_energy}\n")
    print('\n'.join(['\t'.join([str(score) for score in row]) for row in matrix]))
    print(error_list)
    matrix = np.array(matrix)
    savetxt("testing.csv", matrix, fmt="%s")

def run_replicates(number, recep_files, lig_files, recep_names):
    results_list = []
    for index, files in enumerate(zip(recep_files, lig_files, recep_names)):
        for i in range(number):
            print(f"Docking {files[0]} and {files[1]}...")
            v = Vina(sf_name="vina")
            v.set_receptor(files[0])
            v.set_ligand_from_file(files[1])
            v.compute_vina_maps(center=[-17.8626, -4.41595, -15.0678], box_size=[13.89, 12.61, 13.09])
            v.dock(exhaustiveness=8, n_poses=5)
            energies = v.energies()
            energy_list = [row[0] for row in energies]
            min_energy = min(energy_list)
            results_list.append([index, min_energy, files[2]])
    df = pd.DataFrame(results_list, columns= ["run", "binding_energy", "receptor"])
    df.to_csv("final.csv", index=False)

def run_exhaustivness_replicates(receptor_files, ligand_files):
    results_list = []
    x = ""
    receptor = receptor_files[1]
    shortest = ligand_files[1]
    longeest = ligand_files[3]
    ligand_list = [shortest,longeest]
    for exhaustiveness in range(8, 66, 2):
        for ligand in ligand_list:
            for i in range(5):
                v = Vina(sf_name="vina")
                v.set_receptor(receptor)
                v.set_ligand_from_file(ligand)
                v.compute_vina_maps(center=[-17.8626, -4.41595, -15.0678], box_size=[17.94, 12.61, 16.09])
                v.dock(exhaustiveness=exhaustiveness, n_poses=3)
                energies = v.energies()
                energy_list = [row[0] for row in energies]
                min_energy = min(energy_list)
                if ligand == ligand_list[0]:
                    x = "shortest"
                if ligand == ligand_list[1]:
                    x = "longest"
                results_list.append([min_energy, x, exhaustiveness])
    df = pd.DataFrame(results_list, columns=["binding_energy", "length", "exhaustiveness"])
    df.to_csv("exhaustiveness.csv", index = False)



    print(ligand_files)


def main():
    receptor_folder = "/home/sjb176/IRP/pymol/docking_receptors/"
    ligand_folder = "/home/sjb176/IRP/pymol/docking_ligands"
    receptor_files, ligand_files = get_folder_contents(receptor_folder, ligand_folder)
    receptor_names, ligand_names = get_names(receptor_files, ligand_files)
    matrix = initialise_matrix(receptor_names,ligand_names)
    #run_vina(matrix, receptor_files, ligand_files, receptor_names, ligand_names)
    #run_replicates(500, receptor_files, ligand_files, receptor_names)
    run_exhaustivness_replicates(receptor_files, ligand_files)


main()