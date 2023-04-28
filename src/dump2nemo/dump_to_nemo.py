import argparse
import json
import os
import shutil
from pathlib import Path

import soundfile
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("formatter from espnet dump to nemo dump")

    parser.add_argument("--espnet-dump-dir", required=True)
    parser.add_argument("--nemo-wav-dir", default="nemo_wav")
    parser.add_argument("--manifests-dir", default="manifests")
    parser.add_argument("--train-name", default="train_sp")
    parser.add_argument("--dev-name", default="dev1")
    parser.add_argument("--test-name", default="test", nargs="+")
    parser.add_argument("--num_job", type=int, default=-1)
    parser.add_argument("--with-speaker-id", action="store_true")
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


def make_wavscp_dict(line: str, espnet_dump_dir: Path, nemo_wav_dir: Path):
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

    # fileの置き換え (nemoではwavしか受け付けない)
    dirs = str(audio_path).split("/")
    parent = dirs[-2]  # format.*
    filename: str = dirs[-1]  # A01F0019_0047680_0054321.flac
    filename.replace(".flac", ".wav")
    dist_dir = nemo_wav_dir / parent
    dist_dir.mkdir(exist_ok=True)
    new_audio_path = dist_dir / filename
    soundfile.write(str(new_audio_path), data, sr)

    ext_dict = {"audio_filepath": str(new_audio_path), "duration": duration}
    return id, ext_dict


def make_speaker_dict(line: str):
    """
    speaker_idの情報を持った辞書型配列を作る
    並列処理
    """

    id_speaker = line.strip().split(" ")
    id = id_speaker[0]
    speaker = id_speaker[1]
    return id, {"speaker_id": speaker}


def convert_text_id_to_number_iud(data_dict: dict, label_name: str):
    spk_id_set = set(item[label_name] for item in [*data_dict.values()])
    text_to_number = {text_id: number_id for number_id, text_id in enumerate(spk_id_set)}
    return {utt_id: {label_name: text_to_number[item[label_name]]} for utt_id, item in data_dict.items()}


def make_nemo_dump(
    espnet_dump_dir: str,
    nemo_wav_dir: str,
    data_name: str,
    output_dir: str,
    attach_speaker: bool = False,
    job_num: int = -1,
):
    """
    NeMoを学習させるためのdumpファイルを作成します．
    """

    espnet_dump_dir = Path(espnet_dump_dir)
    text_path = espnet_dump_dir / "raw" / data_name / "text"
    wavscp_path = espnet_dump_dir / "raw" / data_name / "wav.scp"
    utt_to_spk_path = espnet_dump_dir / "raw" / data_name / "utt2spk"

    # espnet のTag機能のため
    data_name = data_name.replace("/", "_")
    output_json = Path(output_dir) / f"{data_name}_manifest.json/"
    nemo_wav_dir = Path(nemo_wav_dir)

    assert text_path.exists(), f"{text_path} does not exist"
    assert wavscp_path.exists(), f"{wavscp_path} does not exist"
    if attach_speaker:
        assert utt_to_spk_path.exists(), f"{utt_to_spk_path} does not exist"

    # textのdictを作成
    logger.info("read text file with parallel")
    with open(text_path) as f_text:
        # multi process
        text_dicts = Parallel(n_jobs=job_num)(
            delayed(make_text_dict)(line) for line in tqdm(f_text.readlines(), leave=False)
        )
        text_dicts = dict(text_dicts)

    logger.info("read wav.scp file with parallel")
    with open(wavscp_path) as f_wavscp:
        wavscp_dicts = Parallel(n_jobs=job_num)(
            delayed(make_wavscp_dict)(line, espnet_dump_dir, nemo_wav_dir)
            for line in tqdm(f_wavscp.readlines(), leave=False)
        )
        wavscp_dicts = dict(wavscp_dicts)

    if attach_speaker:
        logger.info("read utt2spk file with parallel")
        with open(utt_to_spk_path) as f_spk_to_txt:
            utt_to_spk_dicts = Parallel(n_jobs=job_num)(
                delayed(make_speaker_dict)(line) for line in tqdm(f_spk_to_txt.readlines(), leave=False)
            )
            utt_to_spk_dicts = dict(utt_to_spk_dicts)
        spk_to_txt_dict = convert_text_id_to_number_iud(utt_to_spk_dicts, "speaker_id")

    # 集計
    dump_list = []
    for id, data in text_dicts.items():
        assert id in wavscp_dicts, f"{id} doesn't exist in text file!!!"
        data.update(wavscp_dicts[id])
        if attach_speaker:
            data.update(spk_to_txt_dict[id])
        dump_list.append(json.dumps(data, ensure_ascii=False) + "\n")

    logger.info("finish to read files. exporting....")
    with open(output_json, "a") as f_json:
        f_json.writelines(dump_list)


def main():
    args = get_args()

    # params
    espnet_dump_dir = Path(args.espnet_dump_dir)
    wav_dir = Path(args.nemo_wav_dir)
    manifests_dir = Path(args.manifests_dir)
    nj = args.num_job

    # ディレクトリ準備
    if manifests_dir.exists():
        shutil.rmtree(manifests_dir)
    manifests_dir.mkdir(parents=True)
    if wav_dir.exists():
        shutil.rmtree(wav_dir)
    wav_dir.mkdir(parents=True)

    # convert処理
    espnet_data_names = [args.train_name, args.dev_name] + args.test_name

    for espnet in espnet_data_names:
        logger.info(f"start format {espnet}")
        nemo_manifest_dir = manifests_dir / espnet
        nemo_wav_dir = wav_dir / espnet

        nemo_manifest_dir.mkdir(parents=True, exist_ok=True)
        nemo_wav_dir.mkdir(parents=True, exist_ok=True)

        # convert
        make_nemo_dump(
            espnet_dump_dir, nemo_wav_dir, espnet, nemo_manifest_dir, attach_speaker=args.with_speaker_id, job_num=nj
        )

    logger.info("finished")


if __name__ == "__main__":
    main()
