"""
an example of adversarial training.
sample code from tf-models/research/adversarial_text

the steps using adv training are as following:
1. calculate loss 
2. use loss to get adv_loss, and add them to get total_loss.
3. use total_loss to optimize the model.

It could be seen that the key is step 2.

total_loss = loss + adv_reg_coeff * adv_loss

Since loss could be got from the classifier_graph or language_model_graph, 
and adv_reg_coeff is a hyper_param,
here we focus on how to get the adv_loss.

there are 3 kinds of adv_losses: 
rp  : random perturbation training
at  : adversarial training, gradient related 
vat : virtual adversarial training, KL_divergence related.
"""

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Adversarial and virtual adversarial training parameters.
flags.DEFINE_float('perturb_norm_length', 5.0,
                   'Norm length of adversarial perturbation to be optimized with validation.')
flags.DEFINE_integer('num_power_iteration', 1, 'The number of power iteration')
flags.DEFINE_float('small_constant_for_finite_diff', 1e-1,
                   'Small constant for finite difference method')
flags.DEFINE_float('adv_reg_coeff', 1.0,
                   'Regularization coefficient of adversarial loss.')


# 1st kind avd_loss
def random_perturbation_loss(embedded, length, loss_fn):
    """Adds noise to embeddings and recomputes classification loss."""
    noise = tf.random_normal(shape=tf.shape(embedded))
    perturb = _scale_l2(_mask_by_length(noise, length), FLAGS.perturb_norm_length)
    return loss_fn(embedded + perturb)

# 2nd kind adv_loss, perturb by gradient 
def adversarial_loss(embedded, loss, loss_fn):
    """Adds gradient to embedding and recomputes classification loss."""
    grad, = tf.gradients(
        loss,
        embedded,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    grad = tf.stop_gradient(grad)
    perturb = _scale_l2(grad, FLAGS.perturb_norm_length)
    return loss_fn(embedded + perturb)

# 3rd kind adv_loss
def virtual_adversarial_loss(logits, embedded, inputs,
                             logits_from_embedding_fn):
    """
    Virtual adversarial loss

    Computes virtual adversarial perturbation by finite difference method and
    power iteration, adds it to the embedding, and computes the 
    <KL divergence between the new logits and the original logits>.

    Args:
        logits: 3-D float Tensor, [batch_size, num_timesteps, m], where m=1 if
                num_classes=2, otherwise m=num_classes.
        embedded: 3-D float Tensor, [batch_size, num_timesteps, embedding_dim].
        inputs: VatxtInput.
        logits_from_embedding_fn: callable that takes embeddings and returns
                                  classifier logits.

    Returns:
        kl: float scalar.
    """
    # Stop gradient of logits. See https://arxiv.org/abs/1507.00677 for details.
    logits = tf.stop_gradient(logits)

    # Only care about the KL divergence on the final timestep.
    weights = inputs.eos_weights

    assert weights is not None

    if FLAGS.single_label:
        indices = tf.stack([tf.range(FLAGS.batch_size), inputs.length - 1], 1)
        weights = tf.expand_dims(tf.gather_nd(inputs.eos_weights, indices), 1)

    # Initialize perturbation with random noise.
    # shape(embedded) = (batch_size, num_timesteps, embedding_dim)
    d = tf.random_normal(shape=tf.shape(embedded))

    # Perform finite difference method and power iteration.
    # See Eq.(8) in the paper http://arxiv.org/pdf/1507.00677.pdf,

    # Adding small noise to input and taking gradient 
    # with respect to the noise corresponds to 1 power iteration.
    for _ in range(FLAGS.num_power_iteration):
        d = _scale_l2(_mask_by_length(d, inputs.length), FLAGS.small_constant_for_finite_diff)

        d_logits = logits_from_embedding_fn(embedded + d)
        kl = _kl_divergence_with_logits(logits, d_logits, weights)  # logits and perturb_logits
        d, = tf.gradients(
            kl,
            d,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        d = tf.stop_gradient(d)

    perturb = _scale_l2(d, FLAGS.perturb_norm_length)
    vadv_logits = logits_from_embedding_fn(embedded + perturb)
    return _kl_divergence_with_logits(logits, vadv_logits, weights)


class Model:
    """
    a classification model
    """
    def __init__(self, loss_fn, adversarial_loss_fn):
        self.loss_fn = loss_fn
        self.adversarial_loss_fn = adversarial_loss_fn  # {rp, at, vat}

    def classifier_graph(self, embedded, labels=None, inputs=None):

        if labels is None:  # for inference mode, no labels for inference data
            logits = self.loss_fn(embedded)
            return logits
        
        # for training mode
        # 1st step: calculate loss 
        logits, loss = self.loss_fn(embedded, labels, return_logits=True)  # loss_fn

        # 2nd step: get adversarial loss
        if self.adversarial_loss_fn != virtual_adversarial_loss:
            adversarial_loss = self.adversarial_loss_fn(embedded, 
                                                        loss, 
                                                        self.loss_fn)  # self.loss_fn(return_logits=False)
        else:
            adversarial_loss = self.adversarial_loss_fn(logits, 
                                                        embedded, 
                                                        inputs, 
                                                        logits_from_embedding)
        adv_loss = (adversarial_loss * 
                    tf.constant(FLAGS.adv_reg_coeff, name='adv_reg_coeff'))
        
        total_loss = loss + adv_loss
        return total_loss
        

def _mask_by_length(t, length):
    """Mask t, 3-D [batch, time, dim], by length, 1-D [batch,]."""
    maxlen = t.get_shape().as_list()[1]

    # Subtract 1 from length to prevent the perturbation from going on 'eos'
    mask = tf.sequence_mask(length - 1, maxlen=maxlen)
    mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
    # shape(mask) = (batch, num_timesteps, 1)
    return t * mask

def _scale_l2(x, norm_length):
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)
    alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(
        tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit

def _kl_divergence_with_logits(q_logits, p_logits, weights):
    """
    Returns weighted KL divergence between distributions q and p.

    Args:
        q_logits: logits for 1st argument of KL divergence shape
                [batch_size, num_timesteps, num_classes] if num_classes > 2, and
                [batch_size, num_timesteps] if num_classes == 2.
        p_logits: logits for 2nd argument of KL divergence with same shape q_logits.
        weights: 1-D float tensor with shape [batch_size, num_timesteps].
                Elements should be 1.0 only on end of sequences

    Returns:
        KL: float scalar.
    """
    # For logistic regression
    if FLAGS.num_classes == 2:
        q = tf.nn.sigmoid(q_logits)
        kl = (-tf.nn.sigmoid_cross_entropy_with_logits(logits=q_logits, labels=q) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logits, labels=q))
        kl = tf.squeeze(kl, 2)

    # For softmax regression
    else:
        q = tf.nn.softmax(q_logits)
        kl = tf.reduce_sum(
            q * (tf.nn.log_softmax(q_logits) - tf.nn.log_softmax(p_logits)), -1)

    num_labels = tf.reduce_sum(weights)
    num_labels = tf.where(tf.equal(num_labels, 0.), 1., num_labels)

    kl.get_shape().assert_has_rank(2)
    weights.get_shape().assert_has_rank(2)

    loss = tf.identity(tf.reduce_sum(weights * kl) / num_labels, name='kl')
    return loss


def loss_fn(embedded, labels=None, inputs=None, return_logits=False):
    """
    returns logits, loss if return_logits else return loss
    return logits if labels is None
    """
    pass

def logits_from_embedding(embedded):
    """
    return only logits
    """
    logits, _ = loss_fn(embedded, return_logits=True)
    return logits
