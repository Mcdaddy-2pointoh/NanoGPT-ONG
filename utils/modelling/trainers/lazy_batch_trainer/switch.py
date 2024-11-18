
def switch(status_dictionary: dict, loss: list = None, previous_segment: str = None):
    """
    Function: Selects the next segment to train and updates the status dictionary with previous segments training information
    Args:
        status_dictionary (dict): A dictionary that keeps track of the training progress over steps
        loss (list): A list of the loss improvement over steps for that segment
        previous_segment (dict): name of the previous segment
    """

    # Validate input types
    if not isinstance(status_dictionary, dict):
        raise TypeError("Argument `status_dictionary` must be of type dict.")

    # Validate status_dictionary keys
    elif not set(list(status_dictionary.keys())).issuperset(set(['pending', 'trained', 'losses'])):
        raise ValueError("Argument `status_dictionary` must contain keys ['pending', 'trained', 'losses']")
    
    # Validate loss
    elif not isinstance(loss, list) and loss is not None:
        raise TypeError("Argument `loss` must be of type list")
    
    # validate previous_segment
    elif not isinstance(previous_segment, str) and previous_segment is not None:
        raise TypeError("Argument `previous_segment` must be of type string")
    
    # Else update the status dictionary and return the new segment and the statust dictionary
    else:

        if loss is not None and previous_segment is not None:
            try:
                status_dictionary = {
                    'pending' : status_dictionary['pending'].remove(previous_segment),
                    'trained' : status_dictionary['trained'].append(previous_segment),
                    'losses' : {**status_dictionary['losses'], previous_segment: loss}
                }
            
            except Exception as e:
                raise ValueError("Could not update `status_dictionary`.") from e
            
        else:
            try:
                status_dictionary = {
                    'pending' : status_dictionary['pending'],
                    'trained' : [],
                    'losses' : {},
                }
            
            except Exception as e:
                raise ValueError("Could not update `status_dictionary`.") from e
            
        # If no segment left to train upon
        if len(status_dictionary['pending']) == 0:
            return status_dictionary, None
        
        # Else return the pending segment
        else:
            return status_dictionary, status_dictionary['pending'][0]