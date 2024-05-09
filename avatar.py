from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
import openai
import google.generativeai as genai
from PIL import Image
import yaml
import os
os.environ['OPENAI_API_KEY'] = 'Your_Api'
client = openai.OpenAI()
gpt4 = ChatOpenAI(model="gpt-4")

GOOGLE_API_KEY = 'Your_API'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro-vision')

#--------------------------------------------

# Ask the place to go.
location = input('''Where do you want to go today? 
    (like Hawaii, Taipei, or Sillicon Valley, etc.) ''')

#--------------------------------------------

class ImageInput(BaseModel):
    prompt: str = Field(description='''The prompt for the image 
        generation like "An young asian female person standing at an office".''')

def generate_image(prompt:str) -> str:
    '''Generate an image for the prompt, and return the filename of the image.'''

    print("Generating image ... for the prompt: ", prompt)

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )

    image_url = response.data[0].url
    print(image_url)

    # download the image
    import requests
    import shutil
    from datetime import datetime

    response = requests.get(image_url, stream=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'output-{timestamp}.png'

    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)

    print(f"Image saved to {filename}")
    return filename

image_generation = StructuredTool.from_function(
    func=generate_image,
    name="image_generation",
    description='''generate an image for the prompt and return the 
        filename of the image.''',
    args_schema=ImageInput,
    return_direct=True,
)

#--------------------------------------------

def analyze_image(filepath:str, prompt:str) -> str:
    '''Analyze an image of the filepath, and return the analyzation of the image.'''

    print("Analyzing the image ... : ", filepath)

    img = Image.open(filepath)

    response = model.generate_content([prompt, img])
    response.resolve()

    analyzed_text = response.text

    print(analyzed_text)

    return analyzed_text

class ProfileCreationInput(BaseModel):
    filepath: str = Field(description='''The file path of the image 
        to be analyzed''')

def create_profile(filepath:str) -> str:
    '''Analyze an image of the filepath, and return the profile of 
       the person in the image.'''

    prompt = '''Create a background story for the person in the picture
        to introduce the persion itself to the user.

        It should include the name of the person, the job, the gender, 
        the age, the nation, etc. based on the situation of the picture.
        '''
    return analyze_image(filepath, prompt)

profile_creation = StructuredTool.from_function(
    func=create_profile,
    name="profile_creation",
    description='''Analyzing an image of the filepath, 
        and return the profile of the person in the image.''',
    args_schema=ProfileCreationInput,
    return_direct=True,
)

#--------------------------------------------

avator_maker_agent = Agent(
    role = 'Avator Maker',
    goal = '''Create a prompt to generate an image of a person who 
        talks with the user in English.
        The prompt must include the age of the person like "young" or "old",
        the region of the person like "asian", "african", "european", and "indian", etc.,
        the gender of the person like "female" or "male", etc.,
        the behaviour of the person like "standing", "smiling", "angry", "sad", and "surprised", etc.,
        and the scene of the image like "an office", "a beach", "a city", "a forest" and "a mountain", etc.
        For example, "An young asian female person standing at an office.".
        And the prompt should be appropriate for the location provided by the user.
        The agent must generate the image for the prompt.''',
    backstory = 'You are an avator maker who creates an image of a person',
    allow_delegation = False,
    verbose = True,
    llm = gpt4,
    tools = [image_generation],
    )


image_generation_task = Task(
    description = f'Create an image of a person at {location}.',
    expected_output = 'A filename of the image.',
    agent = avator_maker_agent,
    human_input = False,
)

crew = Crew(
  agents = [avator_maker_agent],
  tasks = [image_generation_task],
  process = 'sequential',
  verbose = 2
  )

filepath = crew.kickoff()
print('####################')
print(filepath)

image_generation_task = Task (
    description = f'Create an image of a person at {location}.',
    expected_output = 'A filename of the image.',
    agent = avator_maker_agent,
    human_input = False,
    )

crew = Crew(
  agents = [avator_maker_agent],
  tasks = [image_generation_task],
  process = 'sequential',
  verbose = 2
  )

filepath = crew.kickoff()
print('####################')
print(filepath)

#--------------------------------------------

scenario_writer_agent = Agent(
    role = 'Scenario Writer',
    goal = '''Create an attractive story of the person in the picture 
        based on the specification of the task.''',
    backstory = '''You are a creative scenario writer who creates 
        an attractive story of a person from the picture to start 
        the conversation with the user.''',
    allow_delegation = False,
    verbose = True,
    llm = gpt4,
    )

profile_creation_task = Task (
    description = f'''
        Create a profile of the person in the picture to introduce 
        the persion itself to the user.
        the picture is saved on the filepath: '{filepath}'.
        ''',
    expected_output = 'the profile of the person',
    agent = scenario_writer_agent,
    tools = [profile_creation],
    human_input = False,
    )

question_creation_task = Task (
    description = f'''
        Create a question from the person in the picture to start 
        the conversation with the user,
        based on the situation of the picture and the profile of 
        the person generated by the previous task.
        ''',
    expected_output = '''the self introduction of the person in 
        the picture, and the first question''',
    agent = scenario_writer_agent,
    human_input = False,
    )


crew = Crew(
  agents = [scenario_writer_agent],
  tasks = [profile_creation_task, question_creation_task],
  process = 'sequential',
  verbose = 2
  )

result = crew.kickoff()

avator_profile = profile_creation_task.output.raw_output
first_question = question_creation_task.output.raw_output

print('####################')
print('The Avator Profile:')
print(avator_profile)

print('####################')
print('The First Question:')
print(first_question)

#--------------------------------------------



# Save the situation (location, filepath, avator_profile, first_question) to the file

situation = {
    "location": location,
    "filepath": filepath,
    "avator_profile": avator_profile,
    "first_question": first_question
}

with open('situation.yaml', 'w') as file:
    yaml.safe_dump(situation, file)
