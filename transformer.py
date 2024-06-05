transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 100
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    total_batches = len(train_loader)

    for batch_num, batch in enumerate(iterator):
        transformer.train()
        eng_batch, osh_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, osh_batch)
        optim.zero_grad()
        osh_predictions = transformer(eng_batch,
                                     osh_batch,
                                     encoder_self_attention_mask.to(device),
                                     decoder_self_attention_mask.to(device),
                                     decoder_cross_attention_mask.to(device),
                                     enc_start_token=False,
                                     enc_end_token=False,
                                     dec_start_token=True,
                                     dec_end_token=True)
        labels = transformer.decoder.sentence_embedding.batch_tokenize(osh_batch, start_token=False, end_token=True)
        loss = criterian(
            osh_predictions.view(-1, kn_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indices = torch.where(labels.view(-1) == oshikwanyama_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indices.sum()

        loss.backward()
        optim.step()
        train_losses.append(loss.item())

        # Calculate accuracy
        predicted_indices = torch.argmax(osh_predictions.view(-1, kn_vocab_size), dim=1)
        valid_indices = (labels.view(-1) != oshikwanyama_to_index[PADDING_TOKEN])
        accuracy = torch.mean((predicted_indices == labels.view(-1)).float() * valid_indices.float())
        epoch_accuracy += accuracy.item()
        train_accuracies.append(accuracy.item())

        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"English: {osh_batch[0]}")
            print(f"Oshikwanyama Translation: {eng_batch[0]}")
            osh_sentence_predicted = torch.argmax(osh_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in osh_sentence_predicted:
                if idx == oshikwanyama_to_index[END_TOKEN]:
                    break
                predicted_sentence += index_to_oshikwanyama[idx.item()]
            print(f"Oshikwanyama Prediction: {predicted_sentence}")
            print("-------------------------------------------")

    epoch_loss /= total_batches
    epoch_accuracy /= total_batches
    print(f"Epoch {epoch} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}")
