import pandas as pd
import sys
from datetime import datetime
import os
import csv

def parse_file(file_name, model, slo, distrifusion=False):
    print(f"Parsing file: {file_name}")

    try:
        results = pd.read_csv(file_name)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return {
            "slo_rate": 0,
            "avg_latency": 0,
            "goodput": 0,
            "throughput": 0,
        }
    if distrifusion:
        # distrifusion
        status = results.iloc[:, 1].values.tolist()
        resolution = results.iloc[:, 3].values.tolist()
        latencies = results.iloc[:, -2].values.tolist()
        start_time = results.iloc[:, -4].values.tolist()
        end_time = results.iloc[:, -3].values.tolist()
    else:
        # normal
        status = results.iloc[:, 1].values.tolist()
        resolution = results.iloc[:, -3].values.tolist()
        latencies = results.iloc[:, -1].values.tolist()
        start_time = results.iloc[:, 2].values.tolist()
        end_time = results.iloc[:, 3].values.tolist()
    
    if slo == 3:
        scale = 0.6
    elif slo == 5:
        scale = 1.0
    elif slo == 10:
        scale = 2
    else:
        raise NotImplementedError(f"SLO {slo} not supported.")

    # sdxl
    if model == "sdxl":
        ddl = {
            "512": 16.35 * scale,
            "768": 17.5 * scale,
            "1024": 19.31 * scale
        }
    # sd3
    elif model == "sd3":
        ddl = {
            "512": 11 * scale,
            "768": 18 * scale,
            "1024": 30 * scale
        }
    else:
        raise NotImplementedError(f"Model {model} not supported.")
    # print(ddl)

    # SLO, goodput
    slo_num = 0
    total_num = len(resolution)
    for i in range(total_num):
        if float(latencies[i]) <= ddl[str(resolution[i])] and status[i]:
            slo_num += 1
        # else:
        # elif float(latencies[i]) > ddl[str(resolution[i])]:
        #     print(f"{i}th req beak slo, res = {resolution[i]}, latency = {latencies[i]}")

    # avg latency
    total_latency = 0
    for i in range(total_num):
        total_latency += float(latencies[i])

    # throughput
    start = None
    end = None
    for i in range(total_num):
        time1 = datetime.strptime(start_time[i].split(".")[0], "%Y-%m-%d %H:%M:%S")
        time2 = datetime.strptime(end_time[i].split(".")[0], "%Y-%m-%d %H:%M:%S")
        if start is None:
            start = time1
        else:
            if start > time1:
                start = time1
        if end is None:
            end = time2
        else:
            if end < time2:
                end = time2

    time_diff = end - start

    d = {
        "slo_rate": slo_num / total_num,
        "avg_latency": total_latency / total_num,
        "goodput": slo_num / time_diff.total_seconds(),
        "throughput": total_num / time_diff.total_seconds(),
    }

    print(d)
    return d


def parse_folder(folder, model_name, slo, distrifusion=False):
    # iterate all files in the folder
    if distrifusion:
        path = folder + "/client.csv"
    else:
        path = folder + "/request_data.csv"
    d = parse_file(path, model_name, slo, distrifusion)
    return d
    

def parse():
    # init output csv file
    with open("results.csv", "w", newline="") as csvfile:
        fieldnames = ["model", "qps", "slo", "policy", "dp_size", "slo_rate", "avg_latency", "goodput", "throughput"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Figure 12 and 14
        slo = 5
        root_folder = "./results/"
        for model in ["sdxl", "sd3"]:
            if model == "sdxl":
                qps_list = [0.8, 0.9, 1.0, 1.1, 1.2]
            else:
                qps_list = [0.1, 0.2, 0.3, 0.4, 0.5]
            for qps_base in qps_list:
                for policy in ["fcfs_mixed", "orca_resbyres", "esymred", "distrifusion"]:
                    for dp_size in [1, 2, 4, 8]:
                        if policy == "distrifusion" and dp_size == 1:
                            continue
                        qps = qps_base * dp_size
                        folder_path = f"{root_folder}/{model}/{qps:.1f}_{slo}_{policy}_{dp_size}"
                        is_distrifusion = True if policy == "distrifusion" else False
                        d = parse_folder(folder_path, model, slo, distrifusion=is_distrifusion)
                        writer.writerow({
                            "model": model,
                            "qps": qps,
                            "slo": slo,
                            "policy": policy,
                            "dp_size": dp_size,
                            "slo_rate": d["slo_rate"],
                            "avg_latency": d["avg_latency"],
                            "goodput": d["goodput"],
                            "throughput": d["throughput"],
                        })
        
        # Figure 13
        slo = 5
        root_folder = "./results/"
        for model in ["sdxl", "sd3"]:
            if model == "sdxl":
                qps_list = ["8.8_small", "8.8_medium", "8.8_large"]
            else:
                qps_list = ["3.2_small", "3.2_medium", "3.2_large"]
            for qps in qps_list:
                for policy in ["fcfs_mixed", "orca_resbyres", "esymred", "distrifusion"]:
                    for dp_size in [8]:
                        folder_path = f"{root_folder}/{model}/{qps}_{slo}_{policy}_{dp_size}"
                        is_distrifusion = True if policy == "distrifusion" else False
                        d = parse_folder(folder_path, model, slo, distrifusion=is_distrifusion)
                        writer.writerow({
                            "model": model,
                            "qps": qps,
                            "slo": slo,
                            "policy": policy,
                            "dp_size": dp_size,
                            "slo_rate": d["slo_rate"],
                            "avg_latency": d["avg_latency"],
                            "goodput": d["goodput"],
                            "throughput": d["throughput"],
                        })
        
        # Figure 15
        dp_size = 8
        root_folder = "./results/"
        for model in ["sdxl", "sd3"]:
            if model == "sdxl":
                qps_list = [8.8]
            else:
                qps_list = [3.2]
            for qps in qps_list:
                for policy in ["fcfs_mixed", "orca_resbyres", "esymred", "distrifusion"]:
                    for slo in [3, 5, 10]:
                        if policy != "esymred" and slo != 5:
                            # All other policies are SLO-agnostic, we can reuse the slo=5 results
                            folder_path = f"{root_folder}/{model}/{qps}_{5}_{policy}_{dp_size}"
                        else:
                            folder_path = f"{root_folder}/{model}/{qps}_{slo}_{policy}_{dp_size}"
                        is_distrifusion = True if policy == "distrifusion" else False
                        d = parse_folder(folder_path, model, slo, distrifusion=is_distrifusion)
                        writer.writerow({
                            "model": model,
                            "qps": qps,
                            "slo": slo,
                            "policy": policy,
                            "dp_size": dp_size,
                            "slo_rate": d["slo_rate"],
                            "avg_latency": d["avg_latency"],
                            "goodput": d["goodput"],
                            "throughput": d["throughput"],
                        })

if __name__ == "__main__":
    parse()