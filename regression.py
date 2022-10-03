import argparse
from numpy import random
from numpy import array
from sklearn.linear_model import LinearRegression

def parse_args():
    parser = argparse.ArgumentParser(description='Regression')
    parser.add_argument("-n",
                        "--num_data_points",
                        required=True,
                        type=int,
                        help='Number of data points on line')
    args = parser.parse_args()
    if args.num_data_points > 21 or args.num_data_points < 1:
        raise Exception("num_data_points must be between 1 and 21, inclusive")
    return args

def generate_base(num_data_points):
    base_data = []
    for i in range(num_data_points):
        x = i + 10
        base_data.append(x)
    return base_data

def generate_random(num_data_points):
    random_nums = []
    for i in range(num_data_points):
        r_0_to_1 = random.uniform()
        r = (r_0_to_1 * 2) - 1
        random_nums.append(r)
    return random_nums

def create_data_points(base_data, random_nums):
    assert(len(base_data) == len(random_nums))

    y_points = []
    data_points = []
    for i in range(len(base_data)):
        y = base_data[i] + random_nums[i]
        y_points.append(y)
        data_points.append( (base_data[i], y) )

    reshaped_x = array(base_data).reshape(-1, 1)
    return reshaped_x, array(y_points), data_points

def run_regression(x, y):
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    return model, r_sq

def main():
    args = parse_args()
    num_data_points = args.num_data_points
    base_data = generate_base(num_data_points)
    random_nums = generate_random(num_data_points)
    x, y, data_points = create_data_points(base_data, random_nums)
    model, r_sq = run_regression(x, y)

    print(f"Data points: {data_points}\n\n"
          f"Slope: {model.coef_}\n"
          f"Intercept: {model.intercept_}\n"
          f"R squared: {r_sq}")
    return

if __name__ == '__main__':
    main()
