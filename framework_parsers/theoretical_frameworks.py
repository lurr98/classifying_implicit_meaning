import sys, argparse, re, os, tempfile
import xml.etree.ElementTree as ET
from isanlp_rst.parser import Parser
from frame_semantic_transformer import FrameSemanticTransformer
sys.path.append('..')
from prepare_data import read_json, write_json


def prepare_samples(samples_file: str):

    samples_dict = read_json(samples_file)

    samples, revisions, ids = [], [], []

    for idx, sample_info in samples_dict.items():
        ids.append(idx)
        subbed = re.sub("[<>]", "", sample_info["sentence_2"])
        samples.append(subbed)
        rev = re.findall(r"<(.*)>", sample_info["sentence_2"])
        revisions.append(rev[0])

    return samples, revisions, ids


def check_revision_in_framework(revision: str, candidate: str):

    subbed = re.sub("[.,:;\"'?!]", "", candidate)
    if revision.strip().lower() == subbed.strip().lower():
        return True
    else:
        False


def rst_parser(samples: list, revisions: list, ids: list):

    def read_xml(xml_string: str):

        root = ET.fromstring(xml_string)

        segments, relations = [], {}
        for seg in root.findall(".//segment"):
            segments.append({
                "id": seg.attrib["id"],
                "parent": seg.attrib.get("parent"),
                "relname": seg.attrib.get("relname"),
                "text": seg.text.strip() if seg.text else ""
            })

        for rel in root.findall(".//relations"):
            relations[rel.attrib["name"]] = rel.attrib.get("type")
        
        for seg in segments:
            try:
                seg["reltype"] = relations[seg["relname"]]
            except KeyError as e:
                print(f"Key Error: {e}")
        return segments
    # Choose one of the available versions:
    # 'gumrrg', 'rstdt', or 'rstreebank'
    version = 'rstdt'

    parser = Parser(
        hf_model_name='tchewik/isanlp_rst_v3',
        hf_model_version=version,
        cuda_device=-1  # or use -1 if you want CPU
    )
    
    parsed_samples = {}
    for i, sample in enumerate(samples):
        result = parser(sample)
        tree = result['rst'][0]

        # use a temp file
        with tempfile.NamedTemporaryFile(mode="r+", delete=False, suffix=".rs3") as tmp:
            tmp_name = tmp.name
        tree.to_rs3(tmp_name)

        # read back into a string
        with open(tmp_name, encoding="utf-8") as f:
            rs3_string = f.read()

        # cleanup
        os.remove(tmp_name)

        segments = read_xml(rs3_string)
        parsed_samples[ids[i]] = {"rest_tree": [], "revision": revisions[i]}
        for segment in segments:
            if check_revision_in_framework(revisions[i], segment["text"]):
                parsed_samples[ids[i]]["relevant_segment"] = segment
            else:
                parsed_samples[ids[i]]["rest_tree"].append(segment)

    return parsed_samples


def frame_net_parser(samples: list, revisions: list, ids: list):

    frame_transformer = FrameSemanticTransformer()

    frames = {}
    for i, sample in enumerate(samples):
        # @Janosch this is probably the only relevant part for you lol
        result = frame_transformer.detect_frames(sample)

        frames[ids[i]] = {"revision": revisions[i], "rest_tree": []}
        for frame in result.frames:
            for element in frame.frame_elements:
                if check_revision_in_framework(revisions[i], element.text):
                    frames[ids[i]]["relevant_frame"] = {"frame": frame.name, "element": element.name, "text": element.text}
                else:
                    frames[ids[i]]["rest_tree"].append({"frame": frame.name, "element": element.name, "text": element.text})

    return frames


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='tba')
    parser.add_argument('sample_dir', type=str, help='name of the directory containing the samples')
    parser.add_argument('parser', type=str, help='name of the parser to use')
    parser.add_argument('output_dir', type=str, help='name of the directory to store the output')

    args = parser.parse_args()

    for subdir in os.listdir(args.sample_dir):
        topic = subdir[:-6]
        samples, revisions, ids = prepare_samples(os.path.join(args.sample_dir, subdir, "current_samples.json"))

        if args.parser == "rst":
            parsed = rst_parser(samples, revisions, ids)
        elif args.parser == "framenet":
            parsed = frame_net_parser(samples, revisions, ids)

        write_json(parsed, os.path.join(args.output_dir, f"fw_{topic}_study", f"{args.parser}_parsed.json")
