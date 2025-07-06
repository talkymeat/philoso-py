def w(*args):
    with open('fff.txt', 'a') as f:
        f.write('bla\n')
        for arg in args:
            f.write(f'{arg}')
        f.write('\nblah!')
    return args
        