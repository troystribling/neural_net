__all__ = ["import_data", "three_layer_network", "plots"]

def train_network(network, input_data, target_data, epocs=1):
    if epocs < 1:
        print(f"ERROR epocs must be > 1")
        return
    if len(input_data) != len(target_data):
        print(f"ERROR Training Data invalid len(input_data) = {len(input_data)}, len(target_data) = {len(target_data)}")
        return
    train_size = len(input_data)
    print(f"Using {train_size} passes in {epocs} epocs")
    for j in range(epocs):
        print(f"Starting epoc {j+1}")
        for i in range(train_size):
            if i % (train_size/10) == 0:
                print(f"Completed {i} passes {100*i/train_size}% complete")
            input_item = input_data[i]
            target_item = target_data[i]
            network.train(input_item, target_item)
        print(f"Completed epoc {j+1}")
    print("Done")
