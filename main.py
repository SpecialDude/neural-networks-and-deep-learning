from src import mnist_loader, network2


def train_with_nn(training_data, test_data):
    print("Training network with 40 epochs...")
    net = network2.Network([784, 100, 50, 30, 10])
    net.large_weight_initializer()
    net.SGD(training_data, 40, 10, 0.5, evaluation_data=test_data, lmbda=0.1,
            monitor_evaluation_accuracy=True, monitor_evaluation_cost=True,
            monitor_training_accuracy=True, monitor_training_cost=True)


def main():
    data_path = './data/mnist.pkl.gz'
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(data_path)

    train_with_nn(training_data, test_data)



if __name__ == '__main__':
    main()
