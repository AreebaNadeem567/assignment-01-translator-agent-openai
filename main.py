# Import necessary libraries
import os  
from dotenv import load_dotenv 
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled  
import rich 

# Load environment variables from .env file
load_dotenv()

# Disable tracing (debug logging) for cleaner output
set_tracing_disabled(disabled=True)

# Get Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the AsyncOpenAI client with Gemini API key
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  
)

# Create an Agent instance
agent = Agent(
    name="my agent", 
    instructions="You are a translator agent. Convert Urdu text into English and English text into Urdu.",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=client 
    )
)

# Run the agent synchronously with an input text in Urdu
result = Runner.run_sync(
    starting_agent=agent, 
    input="ٹیکنالوجی ہماری زندگی کا ایک اہم حصہ بن چکی ہے۔ آج کل ہر شخص موبائل فون اور انٹرنیٹ کا استعمال کر رہا ہے۔ تعلیم، کاروبار اور رابطے سب کچھ ٹیکنالوجی کے بغیر ناممکن سا لگتا ہے۔ اگر ہم ٹیکنالوجی کو صحیح طریقے سے استعمال کریں تو یہ ہمارے مستقبل کو بہتر بنا سکتی ہے۔"
)

# Print the final translated output in a nicely formatted way
rich.print(result.final_output)
