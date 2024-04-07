import numpy as np

def logistic(x):
    
    """
    Logistic function to calculate probabilities.
    """
    # Check if x is a scalar (float or int)
    # Convert scalar to 1D array
    if np.isscalar(x):
        x = np.array([x])
    
    # Check if x is already a 2D array
    # Convert 1D array to 2D array with a single row
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def calcu_utilities(parameters, data):
    
    """
    Calculate utilities for each alternative based on parameters and data.
    """
    num_alternatives = len(data['AV1'])
    num_params = len(parameters)
    
    #Error handling
    # Assuming each alternative has 7 parameters
    if num_alternatives * 7 != num_params:  
        raise ValueError("Mismatched dimensions between parameters and data")
        
    V = {}
    
    # Iterate over each alternative (assuming AV1, AV2, AV3 are of the same length)
    for alt in range(1, num_alternatives + 1):
        V[alt] = parameters[f'beta01'] + parameters[f'beta1'] * data['X1'][alt - 1] + parameters[f'beta2'] * data['X2'][alt - 1] + parameters[f'beta02'] * data['Sero'][alt - 1] + parameters[f'beta03'] * data['S1'][alt - 1] + parameters[f'betaS1_13'] * data['S1'][alt - 1] * data['AV1'][alt - 1] + parameters[f'betaS1_23'] * data['S1'][alt - 1] * data['AV2'][alt - 1]
    return V

def calcu_probabilities(parameters,data):
    
    """
    calculate probabilities for each alternative.
    """
    V = calcu_utilities(parameters,data)
    probabilities = {}
    
    for alt, v_alt in V.items():
        probabilities[alt] = logistic(v_alt)
    
    return probabilities
    
def save_probabilites(probabilities, filename='output.txt'):
    with open(filename,'w') as file:
        for alt, probs in probabilities.items():
            file.write(f'Alternative:{alt}: {probs}\n')

data = {
    'X1': [2, 1, 3, 4, 2, 1, 8, 7, 3, 2],
    'X2': [8, 7, 4, 1, 4, 7, 2, 2, 3, 1],
    'Sero': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'S1': [3, 8, 4, 7, 1, 6, 5, 9, 2, 3],
    'AV1': [1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
    'AV2': [1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
    'AV3': [1, 1, 0, 0, 1, 1, 1, 1, 1, 1]
}

# Parameters
parameters = {
    'beta01': 0.1, 'beta1': -0.5, 'beta2': -0.4,
    'beta02': 1, 'beta03': 0, 'betaS1_13': 0.33, 'betaS1_23': 0.58
}

try:
    probabilities = calcu_probabilities(parameters, data)
    save_probabilites(probabilities)
    print("Probabilities calculated and saved successfully!")
except ValueError as e:
    print(f"Error: {e}")

#now we will calculate probabilities by call the fuction defined.
probabilities = calcu_probabilities(parameters,data)

save_probabilites(probabilities)





