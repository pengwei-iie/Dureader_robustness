import torch
import args


def evaluate(model, dev_data, device_ids):
    total, losses = 0.0, []

    with torch.no_grad():
        model.eval()
        for batch in dev_data:
            input_ids, input_mask, input_ids_q, input_mask_q, \
            segment_ids, can_answer, start_positions, end_positions = \
                batch.input_ids, batch.input_mask, batch.input_ids_q, batch.input_mask_q, \
                batch.segment_ids, batch.can_answer, \
                batch.start_position, batch.end_position

            input_ids, input_mask, input_ids_q, input_mask_q, \
            segment_ids, can_answer, start_positions, end_positions = \
                input_ids.cuda(), input_mask.cuda(), \
                input_ids_q.cuda(), input_mask_q.cuda(), \
                segment_ids.cuda(), can_answer.cuda(), \
                start_positions.cuda(), end_positions.cuda()


            loss, _, _, _, _ = model(input_ids, input_ids_q, token_type_ids=segment_ids,
                                                    attention_mask=input_mask, attention_mask_q=input_mask_q,
                                                    can_answer=can_answer, start_positions=start_positions,
                                                    end_positions=end_positions)
            loss = loss.mean()
            loss = loss / args.gradient_accumulation_steps
            # print(loss)
            losses.append(loss.item())

        for i in losses:
            total += i
        with open("./log_", 'a') as f:
            f.write("eval_loss: " + str(total / len(losses)) + "\n")

        return total / len(losses)



