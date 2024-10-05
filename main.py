# main.py

import sys
from dataloader import DataLoader

def main():
    # Parse command-line arguments
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'MNIST'
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    # Initialize DataLoader
    data_loader = DataLoader(dataset_name=dataset_name, batch_size=batch_size)

    # # Iterate over data
    # for batch in data_loader:
    #     # Process the batch
    #     print(f"Processing batch of size {len(batch)}")

if __name__ == '__main__':
    main()
