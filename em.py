import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import math

#part 1 (line fit)

# this is the previous version of weighted_lsq when i didn't need weights
# def fit_line(x,y):
#     a = [ [np.sum(np.square(x)), np.sum(x)] , [np.sum(x), 1*x.size] ]
#     b = [np.dot(x,y), np.sum(y)]
#     # no we have a*x = b. it's time to find x
#     x = np.linalg.solve(a,b)
#     return x

# use w = np.ones(x.size) when only one lines needs to be fit
def weighted_lsq(x, y, w):
    a = [ [np.sum(w * np.square(x)), np.sum(w * x)] , [np.sum(w * x), np.sum(w)] ]
    b = [np.sum(w * x * y), np.sum(w * y)]
    # no we have a*x = b. it's time to find x
    x = np.linalg.solve(a,b)
    return x


# x,y is the original data
# this function plots a line according to slope and intercept and adds the x and y's as points
def plot_line(slope, intercept, x, y):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals)
    plt.plot(x,y,'ro')
    plt.show()


# why the weird spacing when printed?
# print(np.arange(start=0, stop=1.05, step=0.05))
def part1_test(part_to_test):
    x = np.arange(start=0, stop=1.05, step=0.05)
    # part 1 i
    if part_to_test == 1:
        y = 2*x + 1
    # part 1 ii
    if part_to_test == 2:
        y = 2*x + 1 + 0.1*np.random.normal(loc=0, scale=1, size=x.size)
    # part 1 iii
    if part_to_test == 3:
        y = (np.abs(x-0.5) < 0.25)*(x+1)+(abs(x-0.5) >= 0.25)*(-x)

    # for the case of 1 line, the Ws are always one
    line = weighted_lsq(x,y,np.ones(x.size))
    a = line[0]
    b = line[1]
    plot_line(a,b,x,y)


 #part1_test(3)

# part 2 EM


def plot2lines(i, line1_params, line2_params, x, y):
    plt.figure()
    plt.subplot(211)
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y1_vals = line1_params[0] * x_vals + line1_params[1]
    y2_vals = line2_params[0] * x_vals + line2_params[1]
    plt.title("iteration " + str(i))
    plt.plot(x_vals, y1_vals)  # plot line in default blue
    plt.plot(x_vals, y2_vals, 'g')  # plot line in green
    plt.plot(x, y, 'ro')  # plot points in red
    plt.xlim([-0.2, 1.2])
    plt.ylim([-1.5, 2.5])
    #plt.show()


def plot_weights(i, weights):
    #plt.figure(2)
    plt.subplot(212)
    #plt.title("weights, iteration " + str(i))
    plt.plot(weights[0], weights[1], 'ro')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xlabel("w1")
    plt.ylabel("w2")
    #plt.show()
    plt.savefig("part2_"+str(i)+".png")


def plot2lines_weights(i, line1_params, line2_params, x, y, weights):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y1_vals = line1_params[0] * x_vals + line1_params[1]
    y2_vals = line2_params[0] * x_vals + line2_params[1]
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(x, y)

    #plt.title("iteration " + str(i))
    axs[0].plot(x_vals, y1_vals)  # plot line in default blue
    axs[0].plot(x_vals, y2_vals, 'g')  # plot line in green
    axs[0].plot(x, y, 'ro')  # plot points in red
    #axs[0].xlim([-0.2, 1.2])
    #axs[0].ylim([-1.5, 2.5])

    axs[1].plot(weights[0], weights[1])

    plt.show()


def get_weight(r1,r2):
    sigma2 = 0.1
    e_r1 = np.exp(-np.square(r1)/sigma2)
    e_r2 = np.exp(-np.square(r2)/sigma2)
    print("r1 = ",r1)
    print("e_r1 = ", e_r1)
    print("e_r2 = ", e_r2)
    # w1 = e_r1/(e_r1 + e_r2)
    # w2 = e_r2/(e_r1 + e_r2)
    w1 = np.divide(e_r1, e_r1 + e_r2, out=np.zeros_like(e_r1), where=(e_r1 + e_r2) != 0)
    w2 = np.divide(e_r2, e_r1 + e_r2, out=np.zeros_like(e_r2), where=(e_r1 + e_r2) != 0)

    return np.array([w1, w2])


def expectation_step(x, y, line1_params,line2_params):
    r1 = line1_params[0]*x + line1_params[1] - y
    r2 = line2_params[0]*x + line2_params[1] - y
    return get_weight(r1, r2)


def maximization_step(x, y, w1, w2):
    line1_params = weighted_lsq(x, y, w1)
    line2_params = weighted_lsq(x, y, w2)
    return [line1_params, line2_params]

# p1,p2 are 2D points
def get_line_from_points(p1,p2):
    slope = (p2[1] - p1[1])/(p2[0] - p1[0])
    b = p2[1] - slope*p2[0]
    return [slope, b]


# find closest point to (x[i],y[i]) from all points
def get_closest_point(x,y,index):
    min_id = 0
    min_dist = math.sqrt((x[0] - x[index])*(x[0] - x[index]) + (y[0] - y[index])*(y[0] - y[index]))
    for i in range(x.size):
        if i == index:
            continue
        dist = math.sqrt((x[i] - x[index])*(x[i] - x[index]) + (y[i] - y[index])*(y[i] - y[index]))
        if dist < min_dist:
            min_dist = dist
            min_id = i
    # plt.plot(x, y, 'o')  # plot points in red
    # plt.plot([ x[index],x[min_id] ],[ y[index],y[min_id] ], 'ro')
    # plt.show()
    return min_id


def em(x, y, n):
    # line1_params = [5, -1]  # a1,b1
    # line2_params = [5, -1]  # a2,b2
    # generate 4 different numbers in range [0,x.size)
    rand_nums = random.sample(range(0, x.size), 4)
    #rand_nums = [2, 7, 6, 12]
    print("random nums = ",rand_nums)
    # init strategy 1
    line1_params = get_line_from_points([ x[rand_nums[0]], y[rand_nums[0]] ], [ x[rand_nums[1]], y[rand_nums[1]] ])
    line2_params = get_line_from_points([x[rand_nums[2]], y[rand_nums[2]]], [x[rand_nums[3]], y[rand_nums[3]]])
    # init strategy 2
    line1_params = np.random.normal(loc=0, scale=1, size=2)
    line2_params = np.random.normal(loc=0, scale=1, size=2)

    #specific example test
    line1_params = [-0.5637512917831672,-0.038763893103619]
    line2_params = [1.5357840771103684, -1.4394713813410815]
    for i in range(n):
        print("iteration ", str(i))
        print("  line 1 = ", line1_params[0], "x + ", line1_params[1])
        print("  line 2 = ", line2_params[0], "x + ", line2_params[1])
        plot2lines(i, line1_params, line2_params, x, y)
        # find weights according to current line params
        weights = expectation_step(x, y, line1_params, line2_params)
        plot_weights(i, weights)
        #plot2lines_weights(i, line1_params, line2_params, x, y,weights)
        print("  w1 = ",weights[0])
        print("  w2 = ", weights[1])
        # update line params according to weights
        line_params = maximization_step(x, y, weights[0], weights[1])
        line1_params = line_params[0]
        line2_params = line_params[1]
    #plot2lines(20, line1_params, line2_params, x, y)


def em_test():
    x = np.arange(start=0, stop=1.05, step=0.05)
    y = (np.abs(x - 0.5) < 0.25) * (x + 1) + (abs(x - 0.5) >= 0.25) * (-x) #+ 0.1*np.random.normal(loc=0, scale=1, size=x.size)
    ##get_closest_point(x,y,2)
    em(x,y,12)

em_test()



