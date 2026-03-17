import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import signal, time, argparse, re, tempfile, spacy, benepar
import xml.etree.ElementTree as ET
from nltk import Tree
from isanlp_rst.parser import Parser
from frame_semantic_transformer import FrameSemanticTransformer
from core.utils import read_json, write_json


# Define custom timeout exception
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function took too long!")

# Set the timeout signal
signal.signal(signal.SIGALRM, timeout_handler)

def run_with_timeout(func, timeout, *args, **kwargs):
    signal.alarm(timeout)  # Set time limit in seconds
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Reset alarm if function completes in time
        return result
    except TimeoutException:
        print("Function timed out!")
        return None


def remove_spaces_wrapper(text: str) -> str:

    cleaned_text = run_with_timeout(remove_spaces, 30, text)

    return cleaned_text

def remove_spaces(text: str) -> str:

    start = time.time()
    # remove spaces before and after "/"
    new_text_slash = re.sub(r'\s*([/])\s*', r'\1', text)
    # Remove spaces inside quotation marks
    new_text_quot = re.sub(r'\s*"\s*([^"]*?)\s*"\s*', r' "\1" ', new_text_slash)
    # Remove spaces inside parentheses
    new_text_paren = re.sub(r'\s*\(\s*([^)]+?)\s*\)\s*', r' (\1) ', new_text_quot)
    # Remove spaces before punctuation
    new_text = re.sub(r'\s+([.,!?;:\'])', r'\1', new_text_paren)
    end = time.time()
    # print(f"Removing spaces took {end-start} seconds")

    # print(f"Took {end-start} seconds.")
    return new_text.strip()


def prepare_samples(samples_dict: dict):

    samples, revisions, ids = [], [], []

    for idx, sample_info in samples_dict.items():
        ids.append(idx)
        subbed = re.sub("[<>]", "", sample_info["sentence_2"])
        samples.append(subbed)
        rev = re.findall(r"<(.*)>", sample_info["sentence_2"])
        revisions.append(rev[0])

    return samples, revisions, ids


def check_revision_in_framework(revision: str, candidate: str):

    sub_contraction = re.sub(" ' ", "'", candidate) 
    subbed = re.sub("[.,:;\"?!]", "", sub_contraction)
    if revision.strip().lower() == subbed.strip().lower():
        return True
    else:
        False


def rst_parser(samples: list, revisions: list, ids: list):

    def read_xml(xml_string: str):

        root = ET.fromstring(xml_string)
        print(root)

        segments, relations = [], {}
        for seg in root.findall(".//segment"):
            segments.append({
                "id": seg.attrib["id"],
                "parent": seg.attrib.get("parent"),
                "relname": seg.attrib.get("relname"),
                "text": seg.text.strip() if seg.text else ""
            })

        # for rel in root.findall(".//relations"):
        #     print(rel)
        #     if "name" in rel.attrib:
        #         relations[rel.attrib["name"]] = rel.attrib.get("type")
        #     # relations[rel.attrib["name"]] = rel.attrib.get("type")
        # 
        # for seg in segments:
        #     print(seg)
        #     try:
        #         seg["reltype"] = relations[seg["relname"]]
        #     except KeyError as e:
        #         print(f"Key Error: {e}")
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

        print(rs3_string)
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


def constituency_parser(samples: list, revisions: list, ids: list):
    
    nlp = spacy.load("en_core_web_lg")

    # Add the Benepar component
    benepar.download('benepar_en3')  # Download the latest English model
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    phrases = {}
    for i, sample in enumerate(samples):
        doc = nlp(sample)
        for sent in doc.sents:
            parse_tree = sent._.parse_string  # constituency parse tree
            tree = Tree.fromstring(parse_tree)  # convert to NLTK Tree

        extracted_phrases, labels = extract_phrases(tree, {})

        try:
            phrases[ids[i]] = {"revision": revisions[i], "rev_phrase": labels[revisions[i]], "phrases": extracted_phrases, "labels": labels}
        except KeyError:
            if revisions[i].startswith("and") or revisions[i].startswith("or"):
                phrases[ids[i]] = {"revision": revisions[i], "rev_phrase": "ADD_COOR", "phrases": extracted_phrases, "labels": labels}
            else:
                phrases[ids[i]] = {"revision": revisions[i], "rev_phrase": None, "phrases": extracted_phrases, "labels": labels}

    return phrases


def extract_phrases(tree: Tree, labels: dict) -> tuple[set, dict]:
    """Extracts all phrases (constituents) from a constituency tree."""
    
    if isinstance(tree, str):  # Base case: Leaf node (word)
        return set(), labels
    
    phrase = remove_spaces_wrapper(" ".join(tree.leaves()))
    phrases = {phrase}  # Join words to form the phrase
    labels[phrase] = tree.label()
    for subtree in tree:
        next_phrase, labels = extract_phrases(subtree, labels)
        phrases.update(next_phrase)  # Recursively extract phrases
        labels[subtree if isinstance(subtree, str) else remove_spaces_wrapper(" ".join(subtree.leaves()))] = 'leaf' if isinstance(subtree, str) else subtree.label()

    phrases = [remove_spaces_wrapper(phrase) for phrase in phrases]
    return phrases, labels

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='tba')
    parser.add_argument('samples', type=str, help='name of the file containing the samples')
    parser.add_argument('parser', type=str, help='name of the parser to use ("rst", "framenet" or "syntax")')
    # parser.add_argument('output_dir', type=str, help='name of the directory to store the output')
    parser.add_argument('-t', '--topics', nargs='?', help='name of the id to topic mapping file')

    args = parser.parse_args()

    base_dir = "/anvme/workspace/v106be21-arr_workspace_december/classifying_implicit_meaning/framework_parsers"

    if args.parser == "rst":
        framework_parser = rst_parser 
    elif args.parser == "framenet":
        framework_parser = frame_net_parser
    elif args.parser == "constituency":
        framework_parser = constituency_parser
    
    samples_dict = read_json(args.samples)

    if args.topics:
        id2topic = read_json(args.topics)

        for topic in set(id2topic.values()):
            samples_dict = {idx: sample for idx, sample in samples_dict.items() if id2topic[idx] == topic}

            samples, revisions, ids = prepare_samples(samples_dict)

            parsed = framework_parser(samples, revisions, ids)

            write_json(parsed, os.path.join(base_dir, args.parser, f"{topic}_{args.parser}_parsed.json"))

    else:
        samples, revisions, ids = prepare_samples(samples_dict)
        
        parsed = framework_parser(samples, revisions, ids)
    
        write_json(parsed, os.path.join(base_dir, args.parser, f"{args.parser}_parsed.json"))
