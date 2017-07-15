def train_network(network, input_data, target_data):
    if len(input_data) != len(target_data):
        print(f"ERROR Training Data invalid len(input_data) = {len(input_data)}, len(target_data) = {len(target_data)}")
        return
    train_size = len(input_data)
    print(f"Using {train_size} passes")
    for i in range(train_size):
        if i % train_size/10 == 0:
            print(f"Completed {i} passes {i/10}% complete")
        input_item = input_data[i]
        target_item = target_data[i]
        network.train(input_item, target_item)
    print("Done")
