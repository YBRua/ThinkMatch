import paddle
# from paddle.nn import DataParallel


def save_model(model, path):
    paddle.save(model.state_dict(), path)


def load_model(model, path):
    model.set_state_dict(paddle.load(path))
