from .probability_space import ProbabilitySpace
from .random_variables import RV

def AssumeIndependent(*args):
    """Construct RVs with the same marginals but defined to be independent.

    Takes RVs defined on separate probability spaces and returns new RVs
    on a joint probability space such that they are mutually independent.

    Parameters
    ----------
    *args : RV
        Any number of RVs, each defined on a different probability space.

    Returns
    -------
    tuple of RV
        RVs with the same marginal distributions as the inputs,
        defined on a common probability space as independent.

    Raises
    ------
    Exception
        If any argument is not an RV.
    Exception
        If any two RVs share the same probability space.

    Examples
    --------
    >>> X = RV(Normal(0, 1))
    >>> Y = RV(Exponential(1))
    >>> X, Y = AssumeIndependent(X, Y)
    >>> (X & Y).sim(10000).plot()
    """

    # Check that none of the RVs are defined on
    # the same probability space.
    for i in range(len(args)):
        if not isinstance(args[i], RV):
            raise Exception(
                "AssumeIndependent(...) can only be "
                "used with RVs, but you passed in a "
                "%s." % type(args[i]).__name__)
        for j in range(i + 1, len(args)):
            if args[i].prob_space == args[j].prob_space:
                raise Exception(
                    "AssumeIndependent(...) can only be "
                    "called on RVs that are initially "
                    "defined on different probability "
                    "spaces."
                    )

    def draw():
        outcome = []
        for arg in args:
            outcome.append(arg.prob_space.draw())
        return outcome
    P = ProbabilitySpace(draw)

    outputs = []
    for i, arg in enumerate(args):
        # i=i forces Python to bind i now
        def _func(x, func=arg.func, i=i):
            return func(x[i])
        outputs.append(RV(P, _func))

    return tuple(outputs)
