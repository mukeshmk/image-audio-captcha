#   implement evaluation function to check the prediction result and view summary report
#   Note:
#   1. Do not scramble while generating the captcha images
#   2. place the file in the project folder where train.py is present
#   3. use the command python evaluate.py --captcha-length 5  --predicted-output stuff.txt

import argparse


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captcha-length', help='Model name to use for classification', type=int)
    parser.add_argument('--predicted-output', help='Model name to use for classification', type=str)
    argument = parser.parse_args()

    if argument.captcha_length is None:
        print("kindly specify the number of characters used to create the captcha")
        exit(1)

    if argument.predicted_output is None:
        print("kindly specify the file that contains predicted results")
        exit(1)

    not_predicted = 0
    success = 0
    failure = 0
    analysis = [0] * (argument.captcha_length - 1)

    with open(argument.predicted_output) as f:
        results = f.readlines()
        results = [x.strip() for x in results]
        not_predicted = len(results)

        for result in results:
            result_array = result.split('.jpg,')
            if result_array[0] == result_array[1]:
                success += 1
            else:
                failure += 1
                r_array = list(result_array[0])
                pre_array = list(result_array[1])
                count = 0
                for index in range(argument.captcha_length - 1):
                    if r_array[index] != pre_array[index]:
                        count += 1
                # analysis[count] += 1

        print("Number of captchas taken for prediction: " + str(not_predicted))
        print("Number succeded in predecting: " + str(success))
        print("Number failed in predecting: " + str(failure))
        accuracy = (success / not_predicted) * 100
        print("Model accuracy is " + str(accuracy) + "%")

        # print("Failure analysis")
        # for index in range(argument.captcha_length):
        #    print(str(index) + " Mismatch count is " + str(analysis[index-1]))


if __name__ == '__main__':
    run()
