import sys
import time


# Progress bar
def print_percent_done(index, total, bar_len=30, title='Please wait'):
    percent_done = (index + 1) / total * 100
    percent_done = round(percent_done, 1)

    done = round(percent_done / (100 / bar_len))
    togo = bar_len - done

    done_str = '=' * int(done)
    togo_str = '.' * int(togo)

    if len(togo_str) >= 1:
        togo_str = ">" + togo_str[1:]

    print(f'{title}: [{done_str}{togo_str}] - {percent_done}%', end='\r')


