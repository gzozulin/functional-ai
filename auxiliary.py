def safe_lambda(lmbda, keys, *args, **kwargs):
    accepted_args = {k: v for k, v in kwargs.items() if k in keys}
    return lmbda(*args, **accepted_args)
