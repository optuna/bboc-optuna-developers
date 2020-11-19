import argparse
import datetime
import itertools
import os
import shutil
import subprocess
import json
import requests

from concurrent.futures import ProcessPoolExecutor, as_completed


FULL_MODELS = [
    "MLP-adam",
    "MLP-sgd",
    "lasso",
    "DT",
    "RF",
    "SVM",
    "ada",
    "kNN",
    "linear",
]
FULL_DATA = ["breast", "digits", "iris", "wine", "boston", "diabetes"]


class BenchmarkLauncher(object):
    def __init__(
        self, db_root, db_id, n_step, n_repeat, n_batch, opt, opt_root, verbose=False
    ):
        self.db_root = db_root
        self.db_id = db_id
        self.n_step = n_step
        self.n_repeat = n_repeat
        self.n_batch = n_batch
        self.opt = opt
        self.opt_root = opt_root
        self.verbose = verbose

    def __call__(self, models, data):
        cmd = [
            "bayesmark-launch",
            "-dir",
            self.db_root,
            "-b",
            self.db_id,
            "-n",
            str(self.n_step),
            "-r",
            str(self.n_repeat),
            "-p",
            str(self.n_batch),
            "-o",
            self.opt,
            "--opt-root",
            self.opt_root,
        ]
        if self.verbose:
            cmd += ["-v"]
        cmd += ["-c"] + models.split()
        cmd += ["-d"] + data.split()
        subprocess.run(cmd, check=True)


def notify_slack(msg, url, channel):
    if url == "":
        url = os.getenv("WEBHOOK_URL", None)
    if channel == "":
        channel = os.getenv("WEBHOOK_SLACK_CHANNEL", None)

    if url is None or channel is None:
        print(msg)
        return

    requests.post(
        url,
        data=json.dumps(
            {
                "channel": channel,
                "text": msg,
                "username": "BBO Challenge Bayesmark Report",
                "link_names": 1,
            }
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Run a benchmark of BBO Challenege.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["small", "large", "custom"],
        help="Size of the benchmark task.",
    )
    parser.add_argument(
        "--optimizer", type=str, required=True, help="Path of the optimizer."
    )
    parser.add_argument("--repeat", type=int, default=10, help="Number of repeat.")
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of jobs that parallelize the benchmark.",
    )
    parser.add_argument(
        "--out", type=str, default="./out", help="Path of the output directory."
    )
    parser.add_argument(
        "--custom-models",
        type=str,
        default=[],
        nargs="+",
        help="Models to be specified with the custom task type.",
    )
    parser.add_argument(
        "--custom-data",
        type=str,
        default=[],
        nargs="+",
        help="Data to be specified with the custom task type.",
    )
    parser.add_argument("--slack-url", type=str, default="", help="Slack Webhook URL")
    parser.add_argument("--slack-channel", type=str, default="", help="Slack channel")
    parser.add_argument("--job-id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--gcs-path", type=str, default="", help="Path of GCS")

    args = parser.parse_args()

    n_step = 16
    n_batch = 8

    task = args.task
    n_repeat = args.repeat

    slack_url = args.slack_url
    slack_channel = args.slack_channel
    gcs_path = args.gcs_path
    job_id = args.job_id

    code_dir = os.path.normpath(args.optimizer)
    opt = os.path.split(code_dir)[1]
    opt_root = os.path.split(code_dir)[0]

    db_root = args.out
    db_id = "run_{}_{}_{}".format(
        task, opt, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    try:
        subprocess.run(["which", "bayesmark-init"], check=True)
    except subprocess.CalledProcessError:
        raise ValueError(
            "Bayesmark has not been installed. Please try: pip install bayesmark."
        )

    os.makedirs(db_root, exist_ok=True)
    if os.path.exists(os.path.join(db_root, db_id)):
        raise ValueError("The DBID {} alrady exists.".format(db_id))

    subprocess.run(["bayesmark-init", "-dir", db_root, "-b", db_id])

    name = "baseline-{}-{}.json".format(n_step, n_batch)
    src = os.path.join(os.path.dirname(__file__), "input", name)
    dist = os.path.join(db_root, db_id, "derived", "baseline.json")
    shutil.copy(src, dist)

    if task == "small":
        models = ["DT", "SVM"]
        data = ["boston", "wine"]
    elif task == "large":
        models = FULL_MODELS
        data = FULL_DATA
    elif task == "custom":
        models = args.custom_models
        data = args.custom_data

        for m in models:
            if m not in FULL_MODELS:
                raise ValueError(
                    "Unknown mdoel is specified in `--custom-models`: {}".format(m)
                )
        for d in data:
            if d not in FULL_DATA:
                raise ValueError(
                    "Unknown data is specified in `--custom-data`: {}".format(d)
                )

        if len(models) == 0 and len(data) == 0:
            raise ValueError(
                "Please specify `--custom-models` or `--custom-data` when using the custom task type."
            )
        if len(models) == 0:
            models = FULL_MODELS
        if len(data) == 0:
            data = FULL_DATA

    else:
        raise ValueError()

    launcher = BenchmarkLauncher(
        db_root=db_root,
        db_id=db_id,
        n_step=n_step,
        n_repeat=n_repeat,
        n_batch=n_batch,
        opt=opt,
        opt_root=opt_root,
    )

    pool = ProcessPoolExecutor(args.parallelism)
    futures = []
    for arg in itertools.product(models, data):
        future = pool.submit(launcher, *arg)
        futures.append(future)

    failure_count = 0
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            failure_count += 1
            print(e)

    cmd = ["bayesmark-agg", "-dir", db_root, "-b", db_id]
    subprocess.run(cmd, check=True)

    cmd = ["bayesmark-anal", "-dir", db_root, "-b", db_id, "-v"]
    anal_output = subprocess.run(cmd, stderr=subprocess.PIPE, check=True)
    anal_text_stderr = anal_output.stderr.decode("utf-8")
    print(anal_text_stderr)
    print("\nFailure count: {}".format(failure_count))

    anal_summary = anal_text_stderr.split('----------\n')[1]
    output_pah = os.path.abspath(os.path.join(db_root, db_id))
    print("\nOutput path: {}".format(output_pah))

    notify_slack(
        f"Job finished: {job_id}\n"
        f"Output Path: {output_pah}\n"
        f"bayesmark-anal: {anal_summary}\n"
        f"failure_count: {failure_count}\n",
        slack_url, slack_channel
    )

    if gcs_path:
        db_folders = os.listdir(db_root)
        print("DB Folders: ", " ".join(db_folders))

        assert len(db_folders) == 1
        cmd = [
            "gsutil",
            "-m",
            "cp",
            "-r",
            os.path.join(db_root, db_folders[0]),
            os.path.join(gcs_path, db_folders[0]),
        ]
        subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True, timeout=10*60,
        )


if __name__ == "__main__":
    main()
