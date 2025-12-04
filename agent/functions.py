from openai import OpenAI
from agent.instruction import SYSTEM_INSTRUCTION
from agent.outputstructure import AIResponse
from typing import AsyncGenerator
from dotenv import load_dotenv
import json
import re

import os
import chromadb
load_dotenv(override=True)


class AgentFunstions:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.chroma_client = chromadb.PersistentClient(path="./vector_db")
        self.collection = self.chroma_client.get_or_create_collection(name="web_chunks",)
        self.db_fetch_size = 5
        self.llm_model = "gpt-4.1" # "gpt-4o-mini"

    async def query_process(self ,user_input: str):
        try:

            # Get result fom vector DB
            results = self.collection.query(query_texts=[user_input], n_results=self.db_fetch_size)

            # LLM parsing
            response = self.client.responses.parse(
                model="gpt-4o-mini", 
                input=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": f"This is the feched content from the DB:-\n\n{results}\n\n User Question:- {user_input}"}
                ],
                temperature=0.3,
                text_format=AIResponse,
                # stream=True
            )

            ai_answer = response.output_parsed.model_dump()
            return {"status": 0, "message": "", "data": ai_answer}


        except Exception as e:
            return {"status": -1, "message": str(e), "data": {}}

    def query_process_stream(self, user_input: str):
        try:
            # 1. Fetch from Vector DB
            results = self.collection.query(
                query_texts=[user_input],
                n_results=self.db_fetch_size
            )

            # 2. Start Stream
            with self.client.responses.stream(
                model= self.llm_model,
                input=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": f"DB Result:\n{results}\n\nUser query: {user_input}"}
                ],
                text_format=AIResponse,
                temperature=0.3,
            ) as stream:
                
                # --- FILTERING LOGIC ---
                buffer = ""
                answer_started = False
                answer_ended = False
                
                for event in stream:
                    if event.type == "response.output_text.delta":
                        chunk = event.delta
                        
                        # If we already finished the answer text, ignore the rest of the stream
                        # (This hides the "cards": [...] JSON syntax from the UI)
                        if answer_ended:
                            continue

                        buffer += chunk

                        # Step A: Look for the start of the answer field
                        if not answer_started:
                            # Regex finds "answer": " (ignoring whitespace)
                            match = re.search(r'"answer"\s*:\s*"', buffer)
                            if match:
                                answer_started = True
                                # Remove the key and start quote from buffer, keep the content
                                buffer = buffer[match.end():]
                        
                        # Step B: Process and stream the answer content
                        if answer_started:
                            i = 0
                            clean_chunk = ""
                            
                            while i < len(buffer):
                                char = buffer[i]
                                
                                # Handle Escape Sequences (e.g., \n, \", \\)
                                if char == '\\':
                                    # If we have a backslash but no next char, wait for next chunk
                                    if i + 1 >= len(buffer):
                                        break 
                                    
                                    next_char = buffer[i+1]
                                    # Convert JSON escapes to real characters for the UI
                                    if next_char == 'n': clean_chunk += '\n'
                                    elif next_char == '"': clean_chunk += '"'
                                    elif next_char == '\\': clean_chunk += '\\'
                                    else: clean_chunk += next_char
                                    i += 2 # Skip \ and the char
                                
                                # Handle End of Answer
                                elif char == '"':
                                    answer_ended = True
                                    i += 1 # Consume the closing quote
                                    break # Stop processing
                                
                                # Handle Normal Characters
                                else:
                                    clean_chunk += char
                                    i += 1
                            
                            # Send the clean text to frontend
                            if clean_chunk:
                                yield json.dumps({"type": "delta", "content": clean_chunk}) + "\n"
                            
                            # Remove processed characters from buffer
                            buffer = buffer[i:]

                # 3. Send Final Structured Result (Cards + Answer)
                final = stream.get_final_response()
                final_json = final.output_parsed.model_dump()
                yield json.dumps({"type": "result", "content": final_json}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"