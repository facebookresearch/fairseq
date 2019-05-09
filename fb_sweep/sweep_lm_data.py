#!/usr/bin/env python3


def set_data_based_on_shortname(args):

    def set_data(fmt, num_shards):
        if num_shards == 0:
            args.data = fmt.format(0)
        else:
            args.data = ':'.join([fmt.format(i) for i in range(num_shards)])

    # mmap datasets
    if args.data == 'CC-NEWS-en.v7.1':
        set_data('/private/home/myleott/data/data-bin/CC-NEWS-en.v7.1/shard{}', 10)
    elif args.data == 'fb_posts':
        set_data('/data/tmp/fb_posts.en.2018-2019.bpe.mmap-bin/shard{}', 100)
    elif args.data == 'fb_posts_gfs':
        set_data('/mnt/vol/gfsai-flash2-east/ai-group/users/myleott/fb_posts/fb_posts.en.2018-2019.bpe.mmap-bin/shard{}', 100)
    # old datasets
    elif args.data == 'CC-NEWS-en.v6':
        set_data('/private/home/myleott/data/data-bin/CC-NEWS-en.v6', 0)
    elif args.data == 'CC-NEWS-en.v9':
        set_data('/private/home/namangoyal/fairseq-py/data-bin/CC-NEWS-en.v9/shard{}', 100)
    elif args.data == 'bookwiki':
        set_data('/private/home/myleott/data/data-bin/bookwiki.10shards/shard{}', 10)
    elif args.data == 'bookwiki_full':
        set_data('/private/home/myleott/data/data-bin/bookwiki-bin', 0)
    elif args.data == 'fb_posts_old':
        set_data('/data/tmp/mono.english.public.2018-2019.shard{}.sents.bpe-bin', 100)
    elif args.data == 'fb_posts_gfs':
        set_data('/mnt/vol/gfsai-flash2-east/ai-group/users/myleott/fb_posts/en/mono.english.public.2018-2019.shard{}.sents.bpe-bin', 100)
    elif args.data == 'wmt19_en_news_docs':
        set_data('/private/home/myleott/data/data-bin/wmt19_en_news_docs/wmt19_en_news_docs.bpe.shard{}', 100)
    else:
        set_data(args.data, 0)

    return args
