def bmw_finder_price_gt_25k(mileage, price):
    if price > 25000:
        return 1
    else:
        return 0


def bmw_finder_price_gt_20k(mileage, price):
    if price > 20000:
        return 1
    else:
        return 0


def bmw_finder_price_gt_cutoff_price(cutoff_price):
    def c(x,p):
        if p > cutoff_price:
            return 1
        else:
            return 0
    return c


def bmw_finder_decision_boundary(mileage,price):
    if price > 21000 - 0.07 * mileage:
        return 1
    else:
        return 0

