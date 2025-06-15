from models.run_xgb import main

if __name__ == "__main__":
    main()
    # data = pd.read_parquet("data/train.parquet")
    # n_row = data.shape[0]
    # data = data.iloc[math.floor(n_row * 0.95) :, :]
    # data.to_parquet("data/train_reduced.parquet")
