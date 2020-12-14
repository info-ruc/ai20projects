import torch
import os
def resave_model(new_path, old_path):
    # 以旧版本格式存储
    torch.save( torch.load(old_path),
                new_path , 
                _use_new_zipfile_serialization = False
    )

if __name__ == '__main__':
    resave_model(os.getcwd() + '/model_resave.pth', os.getcwd() + '/model.pth')