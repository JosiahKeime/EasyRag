supported_file_types = {"text/plain": "supported", 
                        "text/markdown": "supported",
                        "application/pdf": "not supported yet", 
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "supported", }

system_prompt = """You are an assistant for answering questions based on the provided context.
Use only the information in the context to answer the question. If the context does not contain the answer, say you don't know. Always use all relevant information from the context to provide a complete answer.
"""