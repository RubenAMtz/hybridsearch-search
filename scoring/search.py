from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)
import os
import numpy as np
import logging
import pathlib
import time
from hybrid_search import CustomHybridSearchTool
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import openai
from langchain.llms import AzureOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings


def init():
	db_dir = os.getenv("AZUREML_MODEL_DIR", "")
	rd = pathlib.Path(f"{db_dir}")

	load_dotenv()
	openai_key = os.getenv("OPENAI_API_KEY")
	openai.api_type = "azure"
	openai.api_version = "2022-12-01"
	openai.api_base = "https://dademath-test.openai.azure.com/"
	openai.api_key = openai_key
	embeddings = OpenAIEmbeddings(deployment='embeddings-ada-002', model='text-embedding-ada-002', chunk_size=1, openai_api_key=openai_key)

	logging.info(f"Embeddings dir: {db_dir}")

	logging.info(f"Contents of model dir {os.listdir(rd)}")
	logging.info(f"Contents of model dir lvl 0 {os.listdir(rd.parents[0])}")
	logging.info(f"Contents of model dir lvl 1 {os.listdir(rd.parents[1])}")
	logging.info(f"Contents of model dir lvl 1 {os.listdir(rd.parents[2])}")
	logging.info(f"Contents of model dir lvl 1 {os.listdir(rd.parents[2]/'rubenal'/'db_azure'/'leyes_chromadb_512_40_ada_002_recursive')}")
	path_to_embeddings = f"{rd.parents[2]}" / "rubenal" /" db_azure" / "leyes_chromadb_512_40_ada_002_recursive"
	
	db = Chroma(persist_directory=f"{path_to_embeddings}", embedding_function=embeddings, collection_name='abogado_gpt')
	
	logging.info("Embeddings loaded")
	
	openai.api_version = "2022-12-01"
	llm = AzureOpenAI(temperature=0.1, frequency_penalty=0.1, deployment_name='text-davinci-003', streaming=True, max_tokens=600, verbose=True, openai_api_key=openai_key)
	logging.info("LLM loaded")
	
	global hybrid_search
	hybrid_search = CustomHybridSearchTool(llm=llm, db=db, embeddings=embeddings, logger=logging, top_k_wider_search=62, top_k_reranked_search=4, verbose=True)
	logging.info("Hybrid search loaded")

@input_schema(
    param_name="data", param_type=StandardPythonParameterType({
            "inputs": {
                "query": "puedo manejar borracho?",
            }
        })
)
@output_schema(output_type=StandardPythonParameterType( "abc" ))

def run(data):

    logging.info(type(data))
    logging.info(data)
    # create a list of tuples, where each tuple is "source sentence" and "sentence"
    # the source sentence is repeated for each sentence in the list
    # this is the format that the cross-encoder expects
    start = time.time()
    query = data["inputs"]["query"]
    res = hybrid_search.run(query)
    end = time.time()
    logging.info(f"Time elapsed: {end - start}")
    logging.info(f"Output res: {res}")
    logging.info(f"Res dtype {type(res)}")
    return res


# loading the model through the parent folder of the model
# model_path env variable works when there isn't more files that the framework needs for it to load the model. Huggingface requires configuration files on top of the model file.
# to get access to those files, pass the whole 'project' in the deployment stage
# apparently there is a concept of fast tokenizer, a fast tokenizer is a rust implementation of a tokenizer object, this model does not support fast tokenizers, so we need to pass a tokenizer argument "user_fast"=False
# for input_schema, whatever the parameter name is, that is the name of the key in the json that is passed to the run function, for example:
# input_schema(param_name="data", param_type=StandardPythonParameterType({"inputs": {"source_sentence": "puedo manejar borracho?", "sentences": ["manejar borracho", "manejar sobrio"]}}))
# the key in the json is "data", so the run function will receive a json with a key "data" and the value will be the json that is passed to the input_schema function
# {"data": {"inputs": {"source_sentence": "puedo manejar borracho?", "sentences": ["manejar borracho", "manejar sobrio"]}}}