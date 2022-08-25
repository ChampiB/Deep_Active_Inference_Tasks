import glob
import os
import re
from glob import glob
import pandas as pd
import numpy as np


def highlight_max(s):
    maxi = np.nanmax(s[1:5].values)
    mini = np.nanmin(s[1:5].values)
    prefix = np.where(s == np.nanmax(s[1:5].values), "\\textbf{", '')
    suffix = np.where(s == np.nanmax(s[1:5].values), "}", '')
    s = [mid if isinstance(mid, str) else str("{:.2f}".format((mid - mini) / (maxi - mini))) for mid in s]
    return [pre + mid + suf for (pre, mid, suf) in zip(prefix, s, suffix)]


def get_last_line(file):
    try:
        with open(file, "rb") as file:
            # Go to the end of the file before the last break-line
            file.seek(-2, os.SEEK_END)
            # Keep reading backward until you find the next break-line
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
            return file.readline().decode().replace("\n", "")
    except FileNotFoundError:
        print(f"[Warning]: {file} not found.")
        return None


def main():
    # Get all event* runs from logging_dir subdirectories
    logging_dir = '/home/champib/runs'
    csv_dir = "../csv_dir"
    files = [y for x in os.walk(logging_dir) for y in glob(os.path.join(x[0], '*/events.*'))]

    # Extract performance information
    info = []
    for file in files:
        result = re.search(logging_dir + '/(.+?)_ALE/(.+?)_(.+?)/', file)
        model = result.group(1)
        env = result.group(2)
        seed = int(result.group(3))
        reward = get_last_line(f"{csv_dir}/{os.path.basename(file)}.csv")
        reward = None if reward is None else float(reward.split(",")[1])
        info.append((model, env, seed, reward))

    # Create summary table
    df = pd.DataFrame(columns=["Task", "DQN", "CHMM[reward]", "CHMM[efe]", "Random"])
    tasks = []
    for (model, env, seed, reward) in info:
        if env not in tasks:
            tasks.append(env)
            df.at[tasks.index(env), 'Task'] = env
        if model == "hmm":
            df.at[tasks.index(env), 'Random'] = reward
        elif model == "dqn":
            df.at[tasks.index(env), 'DQN'] = reward
        elif model == "chmm" and seed % 3 == 0:
            df.at[tasks.index(env), 'CHMM[reward]'] = reward
        else:
            df.at[tasks.index(env), 'CHMM[efe]'] = reward

    df.to_csv(f"{csv_dir}/summary_table.csv")
    df = df.fillna(value=np.nan)
    df = df.apply(highlight_max, axis=1, result_type="broadcast")
    style = df.style
    style.hide(axis='index')

    print(style.to_latex())


if __name__ == '__main__':
    main()
