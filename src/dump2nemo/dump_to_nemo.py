import json
import os
import argparse
import shutil
from pathlib import Path
import soundfile
import logging
from tqdm import tqdm
from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s:%(levelname)s:[%(filename)s] %(message)s"
)

"""
e.g. 
python dump_to_nemo.py --espnet-dump-dir path/to/dir --train-name train_sp --dev-name dev --test-name test
"""


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("formatter from espnet dump to nemo dump")
    parser.add_argument("--espnet-dump-dir")
    parser.add_argument("--manifests-dir", default="manifests")

    parser.add_argument("--train-name", default="train_sp")
    parser.add_argument("--dev-name", default="dev1")
    parser.add_argument("--test-name", default="test")

    args = parser.parse_args()
    return args


def make_text_dict(line: str):
    """
    textの情報を持った辞書型配列を作る
    並列処理
    """
    id_text = line.strip().split(" ")
    id = id_text[0]
    text = id_text[1]
    return id, {"text": text}


def make_wavscp_dict(line: str, espnet_dump_dir: Path):
    """
    wav.scpの情報を持った辞書型配列を作る
    並列処理
    """
    id_path = line.strip().split(" ")
    id = id_path[0]
    path = id_path[1]
    audio_path = espnet_dump_dir.parent / path

    assert os.path.exists(audio_path), f"{audio_path} doesn't exist"

    data, sr = soundfile.read(audio_path)
    duration = len(data) / sr

    ext_dict = {"audio_filepath": str(audio_path), "duration": duration}
    return id, ext_dict


def make_nemo_dump(espnet_dump_dir: str, data_name: str, output_dir: str):
    """
    NeMoを学習させるためのdumpファイルを作成します．
    """
    espnet_dump_dir = Path(espnet_dump_dir)
    text_path = espnet_dump_dir / f"raw/{data_name}/text"
    wavscp_path = espnet_dump_dir / f"raw/{data_name}/wav.scp"
    output_json = Path(output_dir) / f"{data_name}_manifest.json/"

    assert text_path.exists(), f"{text_path.exists()} does not exist"
    assert wavscp_path.exists(), f"{wavscp_path.exists()} does not exist"

    # textのdictを作成
    logging.info("read text file with parallel")
    with open(text_path) as f_text:
        # multi process
        text_dicts = Parallel(n_jobs=-1)(
                delayed(make_text_dict)(line) for line in tqdm(f_text.readlines())
        )
        text_dicts = dict(text_dicts)

    logging.info("read wav.scp file with parallel")
    with open(wavscp_path) as f_wavscp:
        wavscp_dicts = Parallel(n_jobs=-1)(
            delayed(make_wavscp_dict)(line, espnet_dump_dir)
            for line in tqdm(f_wavscp.readlines())
        )
        wavscp_dicts = dict(wavscp_dicts)

    # 集計
    dump_list = []
    for id, data in text_dicts.items():
        assert id in wavscp_dicts, f"{id} doesn't exist in text file!!!"
        data.update(wavscp_dicts[id])
        dump_list.append(json.dumps(data, ensure_ascii=False)+"\n")

    logging.info("finish to read files. exporting....")
    with open(output_json, "a") as f_json:
        f_json.writelines(dump_list)


def main():
    args = get_args()

    # params
    espnet_dump_dir = Path(args.espnet_dump_dir)
    manifests_dir = Path(args.manifests_dir)

    nemo_train_dir = manifests_dir / "train"
    nemo_dev_dir = manifests_dir / "dev"
    nemo_test_dir = manifests_dir / "test"

    espnet_train_name = args.train_name
    espnet_dev_name = args.dev_name
    espnet_test_name = args.test_name

    if manifests_dir.exists():
        shutil.rmtree(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    nemo_train_dir.mkdir(exist_ok=True)
    nemo_dev_dir.mkdir(exist_ok=True)
    nemo_test_dir.mkdir(exist_ok=True)

    # convert each dump files
    make_nemo_dump(espnet_dump_dir, espnet_train_name, nemo_train_dir)
    make_nemo_dump(espnet_dump_dir, espnet_dev_name, nemo_dev_dir)
    make_nemo_dump(espnet_dump_dir, espnet_test_name, nemo_test_dir)


if __name__ == "__main__":
    main()
