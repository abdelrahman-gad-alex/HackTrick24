import math

import requests
import numpy as np
from LSBSteg import encode
from riddle_solvers import riddle_solvers

api_base_url = '3.70.97.142:5000'
team_id = 'Lpc12eV'


def init_fox(team_id):
    '''
    In this fucntion you need to hit to the endpoint to start the game as a fox with your team id.
    If a sucessful response is returned, you will recive back the message that you can break into chunkcs
      and the carrier image that you will encode the chunk in it.
    '''
    response = requests.post(f'http://{api_base_url}/fox/start', json={"teamId": team_id})
    print(response)
    message = response.json().get('msg')
    print(message)
    carrier_image = np.array(response.json().get('carrier_image'))
    return message, carrier_image

    pass


def generate_message_array(message, image_carrier):
    '''
    In this function you will need to create your own startegy. That includes:
        1. How you are going to split the real message into chunkcs
        2. Include any fake chunks
        3. Decide what 3 chuncks you will send in each turn in the 3 channels & what is their entities (F,R,E)
        4. Encode each chunck in the image carrier  
    '''
    # Split the message into chunks
    num_of_chunks = 3
    chunks_size = math.ceil(len(message) / num_of_chunks)
    message_chunks = [message[i:min(len(message), i + chunks_size)] for i in range(0, len(message), chunks_size)]
    fake_message = "HelloFromTheOtherSid"
    fake_message_chunks = [fake_message[i:min(len(fake_message), i + chunks_size)] for i in
                           range(0, len(fake_message), chunks_size)]

    message_array = [0] * 9  # Create an array of size 9 filled with zeros

    # Insert elements from array1 into specific positions
    message_array[0] = (encode(image_carrier, message_chunks[0])).tolist()
    message_array[4] = (encode(image_carrier, message_chunks[1])).tolist()
    message_array[8] = (encode(image_carrier, message_chunks[2])).tolist()
    #
    # # Insert elements from array2 into remaining positions
    for i in range(1, 4):
        message_array[i] = (encode(image_carrier, fake_message_chunks[i % 3])).tolist()
        message_array[i + 4] = (encode(image_carrier, fake_message_chunks[(i + 4) % 3])).tolist()
    indicator_array = ['R', 'F', 'F', 'F', 'R', 'F', 'F', 'F', 'R']
    return message_array, indicator_array


def get_riddle(team_id, riddle_id):
    '''
    In this function you will hit the api end point that requests the type of riddle you want to solve.
    use the riddle id to request the specific riddle.
    Note that: 
        1. Once you requested a riddle you cannot request it again per game. 
        2. Each riddle has a timeout if you didnot reply with your answer it will be considered as a wrong answer.
        3. You cannot request several riddles at a time, so requesting a new riddle without answering the old one
          will allow you to answer only the new riddle and you will have no access again to the old riddle. 
    '''

    response = requests.post(f'http://{api_base_url}/fox/get-riddle', json={"teamId": team_id, "riddleId": riddle_id})
    input = response.json().get('test_case')
    return input


def solve_riddle(team_id, solution):
    '''
    In this function you will solve the riddle that you have requested. 
    You will hit the API end point that submits your answer.
    Use te riddle_solvers.py to implement the logic of each riddle.
    ''' 

    response = requests.post(f'http://{api_base_url}/fox/solve-riddle', json={"teamId": team_id, "solution": solution})
    total = response.json().get('total_budget')
    print(response.json())
    return total


def send_message(team_id, messages, message_entities=['F', 'E', 'R']):
    '''
    Use this function to call the api end point to send one chunk of the message. 
    You will need to send the message (images) in each of the 3 channels along with their entites.
    Refer to the API documentation to know more about what needs to be send in this api call. 
    '''
    response = requests.post(f'http://{api_base_url}/fox/send-message', json={"teamId": team_id, "messages": messages, "message_entities": message_entities})
    print(response.json())
    pass

def end_fox(team_id):
    '''
    Use this function to call the api end point of ending the fox game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    2. Calling it without sending all the real messages will also affect your scoring fucntion
      (Like failing to submit the entire message within the timelimit of the game).
    '''
    response = requests.post(f'http://{api_base_url}/fox/end-game', json={"teamId": team_id})
    print(response.text)
    return response


def submit_fox_attempt(team_id):
    '''
     Call this function to start playing as a fox. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as a Fox In phase1.
     In this function you should:
        1. Initialize the game as fox 
        2. Solve riddles 
        3. Make your own Strategy of sending the messages in the 3 channels
        4. Make your own Strategy of splitting the message into chunks
        5. Send the messages 
        6. End the Game
    Note that:
        1. You HAVE to start and end the game on your own. The time between the starting and ending the game is taken into the scoring function
        2. You can send in the 3 channels any combination of F(Fake),R(Real),E(Empty) under the conditions that
            2.a. At most one real message is sent
            2.b. You cannot send 3 E(Empty) messages, there should be atleast R(Real)/F(Fake)
        3. Refer To the documentation to know more about the API handling 
    '''
    message, carrier_image = init_fox(team_id)
    message_array, indicator_array = generate_message_array(message, carrier_image)
    riddles_to_solve = ['sec_medium_stegano',
                        'sec_hard', 'problem_solving_easy', 'problem_solving_medium',
                        'problem_solving_hard']
    for i in range(len(riddles_to_solve)):
        riddle=riddles_to_solve[i]
        print(riddle)
        test_case = get_riddle(team_id, riddle)
        try:
            solution = riddle_solvers[riddle](test_case)
        except:
            if riddle == 'cv_easy' or riddle == 'cv_hard' or riddle == 'ml_medium' or riddle == 'problem_solving_hard':
                solution = 0

            if riddle == 'cv_medium' or riddle == 'ml_easy' or riddle == 'sec_hard' or riddle == 'problem_solving_easy':
                solution = []

            if riddle == 'sec_medium_stegano' or riddle == 'problem_solving_medium':
                solution = "0"
        total_budget = solve_riddle(team_id, solution)
        print('\n')
    for i in range(3):
        send_message(team_id, message_array[i*3:i*3+3], indicator_array[i * 3:i * 3 + 3])
    end_fox(team_id)


submit_fox_attempt(team_id)
# response = requests.post(f'http://13.53.169.72:5000/attempts/student', json={"teamId": team_id})
# print(response.json())
