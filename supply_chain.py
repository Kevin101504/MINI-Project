def supply_chain_action(predicted_demand, current_stock):

    if predicted_demand > current_stock:
        print("Increase production & procurement")

    elif predicted_demand < current_stock:
        print("Reduce procurement")

    else:
        print("Maintain operations")