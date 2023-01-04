from torch import nn


class AdaINLoss:

    def __init__(self, enc, style_weight) -> None:
        self.mse = nn.MSELoss()
        self.loss_network = enc
        self.style_weight = style_weight

    # a simple mse for the content loss
    def content_loss(self, realigned_content, pred_feature_last):
        return self.mse(realigned_content, pred_feature_last)

    # mse of the channel-wise mean and std for style loss
    def style_loss(self, style_features, pred_features):
        style_loss = 0
        for s_ft, p_ft in zip(style_features, pred_features):
            B, C = s_ft.shape[0], s_ft.shape[1]
            s_ft, p_ft = s_ft.view(B, C, -1), p_ft.view(B, C, -1)
            mean_loss = self.mse(s_ft.mean(dim=2), p_ft.mean(dim=2))
            std_loss = self.mse(s_ft.std(dim=2), p_ft.std(dim=2))
            style_loss += (mean_loss + std_loss)
        return style_loss


    def calculate_loss(self, style_features, pred_img, realigned_content):
        pred_features = self.loss_network(pred_img)
        content_loss = self.content_loss(realigned_content, pred_features[-1])
        style_loss = self.style_loss(style_features, pred_features)
        return content_loss + (style_loss * self.style_weight)