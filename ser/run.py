import json
import torch
import datetime
import os
import git


def track_run(p, model, save_val_acc):

    now = datetime.datetime.now()

    time_now = now.strftime("%Y:%m:%d_%H:%M")

    if os.path.exists(f'experiments/{p.name}') == False:
        os.mkdir(f'experiments/{p.name}')

    torch.save(model.state_dict(), f"experiments/{p.name}/{time_now}.pth")

    hyperparameters = {
        "name": p.name,
        "epochs": p.epochs,
        "batch_size": p.batch_size,
        "learning_rate": p.learning_rate
    }

    
    final_val = {'final validation accuracy': save_val_acc[-1]}

    best_val_value = min(save_val_acc)
    best_epoch = save_val_acc.index(best_val_value)
    best_val = {
        'best validation accuracy': best_val_value,
        'epoch': best_epoch
        }

    val_loss_list = {'validation accuracies': save_val_acc}

    
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    git_hash = {"Current git commit hash": sha}
    

    with open(f"experiments/{p.name}/{time_now}.json", 'w') as f:
        json.dump(hyperparameters, f)
        f.write("\n")
        json.dump(final_val, f)
        f.write("\n")
        json.dump(best_val, f)
        f.write("\n")
        json.dump(save_val_acc, f)
        f.write("\n")
        json.dump(git_hash, f)
    
