import numpy as np
from scipy.stats import beta

# Algorithm 17.1
def thompson(theta, N):
    # Initialize
    n = len(theta)  # Number of arms
    a = np.ones(n)  # Initial 'a' hyperparameter
    b = np.ones(n)  # Initial 'b' hyperparameter
    X = np.random.random(N)  # Draw from [0,1] to simulate pulls
    traj = np.zeros(N)       # Initial trajectory

    for k in range(N):
        draw = beta.rvs(a,b)  # Thompson sample for all arms
        index = np.argmax(draw)     # Identify arm to pull
        if X[k] < theta[index]:     # If pull is a success
            a[index] += 1     # Update posterior with success
            traj[k] = traj[k-1] + 1 # Update trajectory
        else:
            b[index] += 1     # Update posterior with failure
            traj[k] = traj[k-1]     # Update trajectory
    
    return traj/np.arange(1,N+1)    # Percentage successes


# Problem 17.13
def abtest(thetas, N, m):
    # Get the number of arms, make thetas an np array, and get the remaining number of pulls
    n = len(thetas)
    thetas = np.array(thetas)
    k = N - n*m

    # Generate the index of the arm to pull
    pulls = np.random.random((m,n))
    testarray = np.sum(pulls < thetas, axis=0)
    index = np.argmax(testarray / m)

    # Generate the rest of the pulls and return the estimated success rate
    pulls = np.random.random(k)
    successes = np.sum(pulls < thetas[index])
    return (np.sum(testarray) + successes) / N

def display13():
    # Set the parameters
    thetas = [.1,.2,.3,.4,.5]
    N = 5000
    m = 100

    # Print the results
    print("\n--Problem 17.13--")
    print("A/B testing: " + str(abtest(thetas,N, m)))
    print("Thompson sampling: " + str(thompson(thetas,N)[-1])+ "\n")


# Problem 17.14
def abtest2(thetas, N, m):
    # Get the number of arms, make thetas an np array, and get the remaining number of pulls
    n = len(thetas)
    thetas = np.array(thetas)
    indices = np.arange(n)

    # Randomly choose from the thetas m times
    draws = np.random.choice(indices, m)
    list = np.random.random(m)
    successes = list < thetas[draws]

    # Initialize the a and b lists
    a = np.zeros(n)
    b = np.zeros(n)

    # Update the a and b lists
    for k, index in enumerate(draws):
        if successes[k]:
            a[index] += 1
        b[index] += 1

    # Set infinities to zero and get the index of which theta has the highest success rate
    ratios = a/b
    ratios[np.isinf(ratios)] = 0
    k = np.argmax(a/ b)

    # Generate the rest of the pulls and concatenate the successes
    pulls = np.random.random(N - m)
    phase2success = pulls < thetas[k]
    successes = np.concatenate((successes, phase2success))
    
    # Return the cumulative success rate
    return np.cumsum(successes) / np.arange(1,N+1)

# Given thetas of .1, .3, and .6, we compare all three methods using same parameters
def display14():
    # Set the parameters
    thetas = [.1,.2,.3,.4,.5]
    N = 5000
    m = 100

    # Print the results
    print("\n--Problem 17.14--")
    print("Regular A/B testing: " + str(abtest(thetas,N, m)))
    print("Alternative A/B testing: " + str(abtest2(thetas,N, m)[-1]))
    print("Thompson sampling: " + str(thompson(thetas,N)[-1]) + "\n")


# Problem 17.15
def newassement(thetas, N, m, beta):
    # First asses thompson sampling and then get A/B testing with m pulls
    thomp = thompson(thetas,N) * np.arange(1,N+1)
    ab = abtest2(thetas,N,m) * np.arange(1,N+1)
    
    # Calculate the rewards for each method and return them
    thompreward = np.sum([thomp[i-1] * (beta ** i) for i in range(1,N+1)])
    abtestreward = np.sum([ab[i-1] * (beta ** i) for i in range(1,N+1)])
    return thompreward, abtestreward

def experiment(thetas, N, m, beta, trials):
    # Initialize the t and ab values
    t = 0
    ab = 0

    # Repeat the experiment trials times
    for i in range(trials):
        tval, abval = newassement(thetas, N, m, beta)
        t += tval
        ab += abval

    # Return the average reward
    return t/trials, ab/trials

def display15():
    # Set the parameters and run the experiment
    thetas = [.1,.2,.3,.4,.5]
    N = 1000
    m = 100
    beta = .9
    trials = 100
    thompson, abtest = experiment(thetas, N, m, beta, trials)

    # print the results
    print("\n--Problem 17.15--")
    print("Thompson sampling average utility: " + str(thompson))
    print("A/B testing average utility: " + str(abtest) + "\n")


# Problem 17.16
def altthompson(J, theta, N):
    # Initialize n, a, and b
    n = len(theta)
    a = np.ones(n)
    b = np.ones(n)

    # Initialize the trajectory and random draws
    X = np.random.random(N)
    traj = np.zeros(N)

    # Loop through the number of pulls and sample from the beta distribution
    for k in range(N):
        draw = beta.rvs(a,b)

        # find the argmax of expected value and if the pull is a success update a and the trajectory
        index = np.argmax(draw * J)
        if X[k] < theta[index]:
            a[index] += 1
            traj[k] = traj[k-1] + J[index]

        # If the pull is a failure update b and the trajectory
        else:
            b[index] += 1
            traj[k] = traj[k-1]
    
    # Return the trajectory
    return traj/np.arange(1,N+1)

def display16():
    # Set the parameters
    J = [6,2,1,1,1]
    theta = [.1,.2,.3,.4,.5]
    N = 10000

    # Print the results
    print("\n--Problem 17.16--")
    print("Thompson sampling avg ex. return (J rewards): " + str(altthompson(J, theta, N)[-1]) + "\n")

# Run the functions
display13()
display14()
display15()
display16()