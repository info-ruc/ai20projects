import os
import sys
os.chdir('/home/peitian_zhang/Codes/NR')
sys.path.append('/home/peitian_zhang/Codes/NR')

import torch
import torch.optim as optim
from datetime import datetime
from torchtext.vocab import FastText
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.MIND import MIND_iter,MIND_map
from utils.utils import getLoss,constructBasicDict,run_eval,run_train
from models.baseline import GCAModel

if __name__ == "__main__":
    
    hparams = {
        'mode':sys.argv[1],
        'epochs':int(sys.argv[2]),
        'name':'baseline',
        'batch_size':100,
        'title_size':20,
        'his_size':100,
        'npratio':4,
        'dropout_p':0.2,
        'query_dim':200,
        'embedding_dim':300,
        'transformer_dim':16,
        'head_num':16,
        'kernel_num':11,
        'metrics':'group_auc,ndcg@5,ndcg@10,mean_mrr',
        'device':'cuda:0',
        'attrs': ['title'],
    }

    news_file_train = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_train/news.tsv'
    news_file_test = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_dev/news.tsv'
    behavior_file_train = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_train/behaviors.tsv'
    behavior_file_test = '/home/peitian_zhang/Data/MIND/MIND'+hparams['mode']+'_dev/behaviors.tsv'
    save_path = 'models/model_params/{}_{}_{}'.format(hparams['name'],hparams['mode'],hparams['epochs']) +'.model'

    if not os.path.exists('data/dictionaries/vocab_{}_{}_{}.pkl'.format(hparams['mode'],'train','_'.join(hparams['attrs']))):
        constructBasicDict(news_file_train,behavior_file_train,hparams['mode'],'train',hparams['attrs'])
    if not os.path.exists('data/dictionaries/vocab_{}_{}_{}.pkl'.format(hparams['mode'],'test','_'.join(hparams['attrs']))):
        constructBasicDict(news_file_test,behavior_file_test,hparams['mode'],'test',hparams['attrs'])

    device = torch.device(hparams['device']) if torch.cuda.is_available() else torch.device("cpu")

    dataset_train = MIND_map(hparams=hparams,mode='train',news_file=news_file_train,behaviors_file=behavior_file_train)
    dataset_test = MIND_iter(hparams=hparams,mode='test',news_file=news_file_test,behaviors_file=behavior_file_test)

    vocab_train = dataset_train.vocab
    embedding = FastText('simple',cache='.vector_cache')
    vocab_train.load_vectors(embedding)

    vocab_test = dataset_test.vocab
    vocab_test.load_vectors(embedding)

    loader_train = DataLoader(dataset_train,batch_size=hparams['batch_size'],shuffle=True,pin_memory=True,num_workers=3,drop_last=True)
    loader_test = DataLoader(dataset_test,batch_size=hparams['batch_size'],pin_memory=True,num_workers=0,drop_last=True)

    writer = SummaryWriter('data/tb/{}/{}/{}/'.format(hparams['name'], hparams['mode'], datetime.now().strftime("%Y%m%d-%H")))

    gcaModel = GCAModel(vocab=vocab_train,hparams=hparams).to(device)
    gcaModel.train()

    if gcaModel.training:
        print("training...")
        loss_func = getLoss(gcaModel)
        optimizer = optim.Adam(gcaModel.parameters(),lr=0.001)
        gcaModel = run_train(gcaModel,loader_train,optimizer,loss_func,writer,epochs=hparams['epochs'], interval=10)

    gcaModel.eval()
    gcaModel.vocab = vocab_test
    gcaModel.npratio = -1

    run_eval(gcaModel,loader_test)

    gcaModel.npratio = 4
    torch.save(gcaModel.state_dict(), save_path)