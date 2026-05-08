import argparse
import torch
import config
import models

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--config_format', default='base', choices=config.registered_formats())

args, rem_args = parser.parse_known_args([
    '--config_format', 'base',
    '--model', 'tak_custom_cot',
    '--n_embd', '768',
    '--n_head', '12',
    '--n_layer', '27',
    '--n_repeat', '5',
    '--n_layer_begin', '2',
    '--n_layer_end', '1',
    '--sequence_length', '256',
    '--dropout', '0.0',
    '--dataset', 'owt2',
    '--device', 'cuda:0',
])

args = config.parse_args_with_format(
    format=args.config_format,
    base_parser=parser,
    args=rem_args,
    namespace=args,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.device = device

print("Building model:", args.model)
model = models.make_model_from_args(args).to(device)
model.eval()

x = torch.randint(
    low=0,
    high=args.vocab_size,
    size=(2, args.sequence_length),
    device=device,
    dtype=torch.long,
)

print("Input shape:", x.shape)

with torch.no_grad():
    out = model(x, get_logits=True)

print("Forward ok")
print("Logits shape:", None if out["logits"] is None else out["logits"].shape)
print("Diag metrics:", None if out["diag_metrics"] is None else out["diag_metrics"].keys())
print("sim_of_xs:", out["sim_of_xs"])
print("var_into:", out["var_into"])
print("var_outof:", out["var_outof"])
